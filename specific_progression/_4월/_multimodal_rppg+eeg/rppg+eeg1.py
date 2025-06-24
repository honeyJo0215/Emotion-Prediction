#!/usr/bin/env python3
# multimodal_diffusion_emotion.py
# -------------------------------------------------------------------
# Multimodal emotion recognition with EEG + rPPG using diffusion-based
# branches and Cross-Modality Gated Attention Fusion (forget-gate)
# -------------------------------------------------------------------

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. Configuration & GPU setup
# =============================================================================
RESULT_DIR = "/home/bcml1/sigenv/_4월/_diffusion/EEG_rPPG_multimodal"
os.makedirs(RESULT_DIR, exist_ok=True)

def limit_gpu_memory(memory_limit_mib=10000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
            )
            print(f"🔧 GPU memory limited to {memory_limit_mib} MiB.")
        except RuntimeError as e:
            print(f"⚠️ GPU config error: {e}")
    else:
        print("ℹ️ No GPU detected; running on CPU.")

limit_gpu_memory(10000)

# =============================================================================
# 2. Diffusion denoising block: Bidirectional SSM + Hourglass + fusion scale
# =============================================================================
class BidirectionalSSMLayer(layers.Layer):
    def __init__(self, units, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.fwd_conv = layers.Conv1D(1, kernel_size, padding='causal', name="ssm_fwd_conv")
        self.bwd_conv = layers.Conv1D(1, kernel_size, padding='causal', name="ssm_bwd_conv")

    def call(self, x, training=False):
        # x: (batch, time, channels) or (batch, features)
        seq = tf.expand_dims(x, axis=-1)  # (..., 1)
        fwd = self.fwd_conv(seq)
        rev = tf.reverse(seq, axis=[1])
        bwd = self.bwd_conv(rev)
        bwd = tf.reverse(bwd, axis=[1])
        out = fwd + bwd
        return tf.squeeze(out, axis=-1)

class DiffuSSMLayer(layers.Layer):
    def __init__(self, model_dim, hidden_dim, cond_dim=None, alpha_init=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization()
        self.cond_mlp = layers.Dense(2 * model_dim) if cond_dim else None
        self.hg_down = layers.Dense(hidden_dim, activation='swish', name="hg_down")
        self.ssm     = BidirectionalSSMLayer(hidden_dim, kernel_size=3)
        self.hg_up   = layers.Dense(model_dim, name="hg_up")
        self.alpha   = self.add_weight("alpha", shape=(), initializer=tf.constant_initializer(alpha_init), trainable=True)

    def call(self, x, cond=None, training=False):
        x_ln = self.norm(x)
        if self.cond_mlp is not None and cond is not None:
            gamma, beta = tf.split(self.cond_mlp(cond), 2, axis=-1)
            x_ln = gamma * x_ln + beta
        h = self.hg_down(x_ln)
        h = self.ssm(h)
        h = self.hg_up(h)
        return x + self.alpha * h

def add_diffusion_noise(x, stddev=0.05):
    return x + tf.random.normal(tf.shape(x), stddev=stddev)

# =============================================================================
# 3. Branch builders: EEG (2D CNN) and rPPG (1D CNN)
# =============================================================================
def build_eeg_branch(input_shape=(4,200,8), noise_std=0.05):
    inp = Input(shape=input_shape, name="EEG_Input")
    x = layers.Permute((2,3,1))(inp)  # (200,8,4)
    x = add_diffusion_noise(x, stddev=noise_std)
    x = layers.Conv2D(16,3,activation='relu',padding='same')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)  # (64,)
    x = add_diffusion_noise(x, stddev=noise_std)
    x = DiffuSSMLayer(model_dim=64, hidden_dim=64)(x)
    return models.Model(inp, x, name="EEG_Branch")

def build_rppg_branch(input_length=200, noise_std=0.05):
    inp = Input(shape=(input_length,1), name="rPPG_Input")
    x = add_diffusion_noise(inp, stddev=noise_std)
    x = layers.Conv1D(16,3,activation='relu',padding='same')(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64,3,activation='relu',padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)  # (64,)
    x = add_diffusion_noise(x, stddev=noise_std)
    x = DiffuSSMLayer(model_dim=64, hidden_dim=64)(x)
    return models.Model(inp, x, name="rPPG_Branch")

# =============================================================================
# 4. Cross-Modality Gated Attention Fusion (EEG ↔ rPPG)
# =============================================================================
class CrossModalGatedAttention(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.Wq = layers.Dense(dim, use_bias=False, name="Wq")
        self.Wk = layers.Dense(dim, use_bias=False, name="Wk")
        self.Wm = layers.Dense(dim, name="Wm")
        self.Wf = layers.Dense(dim, name="Wf")
        self.bf = self.add_weight("bf", shape=(dim,), initializer="zeros")
        self.relu = layers.ReLU()

    def call(self, z_eeg, z_rppg):
        # 1) Cross-attention: rppg queries, eeg keys/values
        Q = self.Wq(z_rppg)                                    # (batch, dim)
        K = self.Wk(z_eeg)                                     # (batch, dim)
        attn_weights = tf.nn.softmax(tf.reduce_sum(Q * K, axis=-1, keepdims=True) / tf.math.sqrt(tf.cast(K.shape[-1],tf.float32)), axis=1)
        A = attn_weights * tf.expand_dims(z_eeg, axis=1)       # (batch, time?, dim) but here dim vectors
        A = tf.reduce_sum(A, axis=1)                           # (batch, dim)

        # 2) Forget gate: f = σ([A ⊕ z_rppg] Wf + bf)
        concat = tf.concat([A, z_rppg], axis=-1)
        f = tf.sigmoid(self.Wf(concat) + self.bf)

        # 3) Filtered interaction: h = ReLU(z_rppg + (A Wm) * f)
        m = self.Wm(A)
        h = self.relu(z_rppg + m * f)
        return h

# =============================================================================
# 5. Data loading
# =============================================================================
def load_eeg_features(dir_path):
    X, Y, subjects, win_ids = [], [], [], []
    for fname in os.listdir(dir_path):
        if not fname.endswith(".npy"): continue
        match_lbl = re.search(r'label(\d+)', fname)
        match_sub = re.search(r'subject(\d+)', fname)
        if not match_lbl or not match_sub: continue
        lbl = int(match_lbl.group(1))
        if lbl == 4:  # exclude label 4
            continue
        sub = match_sub.group(1)
        data = np.load(os.path.join(dir_path, fname))  # shape (4,T,8)
        T = data.shape[1]
        nwin = T // 200
        for i in range(nwin):
            X.append(data[:, i*200:(i+1)*200, :])
            Y.append(lbl)
            subjects.append(sub)
            win_ids.append(f"{fname}_win{i}")
    return (np.array(X), np.array(Y), np.array(subjects), np.array(win_ids))

def load_rppg_features(base_dir, win_length=200):
    X, subjects, win_ids = [], [], []
    for d in sorted(os.listdir(base_dir)):
        sub_dir = os.path.join(base_dir, d)
        if not os.path.isdir(sub_dir): continue
        sub = re.sub(r'\D+', '', d)  # extract digits
        for fname in os.listdir(sub_dir):
            if not fname.endswith(".npy"): continue
            sig = np.load(os.path.join(sub_dir, fname))  # e.g. shape (8064,)
            nwin = len(sig) // win_length
            for i in range(nwin):
                window = sig[i*win_length:(i+1)*win_length]
                X.append(window)
                subjects.append(sub)
                win_ids.append(f"{d}/{fname}_win{i}")
    X = np.array(X)[..., np.newaxis]  # shape (N, win_length,1)
    return (X, np.array(subjects), np.array(win_ids))

# =============================================================================
# 6. Main training loop: intra-subject
# =============================================================================
if __name__ == "__main__":
    # --- Load modalities ---
    EEG_DIR  = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
    RPPG_DIR = "/home/bcml1/sigenv/_4월/_rppg/minipatch_rppg"
    X_eeg, Y, subj_eeg, id_eeg = load_eeg_features(EEG_DIR)
    X_rppg, subj_rppg, id_rppg = load_rppg_features(RPPG_DIR, win_length=200)

    # --- Align EEG & rPPG by subject and window index ---
    # build mapping from (sub, win#) → rPPG window
    map_r = {f"{subj}/{win.split('_win')[1]}": i for i,(subj, win) in enumerate(zip(subj_rppg, id_rppg))}
    pairs_idx = []
    for i, (sub, win) in enumerate(zip(subj_eeg, id_eeg)):
        key = f"{sub}/{win.split('_win')[1]}"
        if key in map_r:
            pairs_idx.append((i, map_r[key]))
    idx_eeg, idx_rppg = zip(*pairs_idx)
    X_eeg = X_eeg[list(idx_eeg)]
    X_rppg = X_rppg[list(idx_rppg)]
    Y      = Y[list(idx_eeg)]
    subj   = subj_eeg[list(idx_eeg)]

    print(f"🔄 Paired {len(Y)} samples across EEG & rPPG")

    unique_subj = np.unique(subj)
    for s in unique_subj:
        mask = (subj == s)
        Xe, Xr = X_eeg[mask], X_rppg[mask]
        Ye = Y[mask]
        if len(Ye) < 10:
            continue  # skip subjects with too few samples

        # remap labels to 0…C-1
        ul = np.unique(Ye); lm = {v:i for i,v in enumerate(ul)}
        y = np.array([lm[v] for v in Ye])
        nc = len(ul)

        # train/val/test split (stratified)
        Xe_tr, Xe_tmp, Xr_tr, Xr_tmp, y_tr, y_tmp = train_test_split(
            Xe, Xr, y, test_size=0.3, random_state=42, stratify=y)
        Xe_val, Xe_te, Xr_val, Xr_te, y_val, y_te = train_test_split(
            Xe_tmp, Xr_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

        # augment training set for class balance
        cnts = np.bincount(y_tr); maxc = cnts.max()
        Xe_aug, Xr_aug, y_aug = [], [], []
        for lbl, c in enumerate(cnts):
            idxs = np.where(y_tr == lbl)[0]
            for _ in range(maxc - c):
                i = np.random.choice(idxs)
                Xe_aug.append(add_diffusion_noise(Xe_tr[i],0.05).numpy())
                Xr_aug.append(add_diffusion_noise(Xr_tr[i],0.05).numpy())
                y_aug.append(lbl)
        if Xe_aug:
            Xe_tr = np.concatenate([Xe_tr, np.array(Xe_aug)], axis=0)
            Xr_tr = np.concatenate([Xr_tr, np.array(Xr_aug)], axis=0)
            y_tr  = np.concatenate([y_tr,  np.array(y_aug)], axis=0)

        # Data pipelines
        train_ds = tf.data.Dataset.from_tensor_slices(((Xe_tr, Xr_tr), y_tr))\
                        .shuffle(1000).batch(16)
        val_ds   = tf.data.Dataset.from_tensor_slices(((Xe_val, Xr_val), y_val)).batch(16)
        test_ds  = tf.data.Dataset.from_tensor_slices(((Xe_te, Xr_te), y_te)).batch(16)

        # Build branches & fusion
        eeg_branch  = build_eeg_branch(input_shape=(4,200,8), noise_std=0.05)
        rppg_branch = build_rppg_branch(input_length=200, noise_std=0.05)
        cmga_layer  = CrossModalGatedAttention(dim=64)

        # Inputs
        in_eeg  = Input(shape=(4,200,8), name="EEG_In")
        in_rppg = Input(shape=(200,1), name="rPPG_In")
        fe = eeg_branch(in_eeg)      # (batch,64)
        fr = rppg_branch(in_rppg)    # (batch,64)
        fused = cmga_layer(fe, fr)   # (batch,64)

        # Classifier
        x = layers.Dense(128, activation='relu')(fused)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(nc, activation='softmax')(x)

        model = models.Model([in_eeg, in_rppg], out, name=f"Subj_{s}_EEG_rPPG")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        print(f"\n▶▶ Training subject {s} (labels: {ul})")
        history = model.fit(train_ds,
                            epochs=500,
                            validation_data=val_ds,
                            callbacks=[EarlyStopping('val_accuracy', patience=50, restore_best_weights=True)],
                            verbose=2)

        # Save curves & evaluation
        subj_dir = os.path.join(RESULT_DIR, f"subj{s}")
        os.makedirs(subj_dir, exist_ok=True)
        # Loss/Acc curves
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend(); plt.title('Loss')
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='train acc')
        plt.plot(history.history['val_accuracy'], label='val acc')
        plt.legend(); plt.title('Accuracy')
        plt.savefig(os.path.join(subj_dir, "training_curves.png"))
        plt.close()

        # Test
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"✅ Subject {s}: Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
        y_pred = np.argmax(model.predict(test_ds), axis=1)
        cm = confusion_matrix(y_te, y_pred)
        report = classification_report(y_te, y_pred)
        # Save metrics
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title("Confusion Matrix"); plt.savefig(os.path.join(subj_dir,"cm.png")); plt.close()
        with open(os.path.join(subj_dir,"report.txt"), "w") as f:
            f.write(report)
        model.save(os.path.join(subj_dir, "model.keras"))

    print("🎯 All subjects done.")
