#!/usr/bin/env python3
# multimodal_diffusion_eeg_rppg.py
# -------------------------------------------------------------------
# Multimodal emotion recognition: EEG (freq + channel branches)
#  + rPPG, using diffusion blocks and Cross-Modality Gated Attention
# -------------------------------------------------------------------

import os
import re
import numpy as np
from scipy.interpolate import interp1d
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
RESULT_DIR = "/home/bcml1/sigenv/_4월/_multimodal_rppg+eeg/mult2"
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
        # x: (batch, features) or (batch, time, channels)
        seq = tf.expand_dims(x, axis=-1)
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
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.constant_initializer(alpha_init),
            trainable=True
        )
        #self.alpha   = self.add_weight("alpha", shape=(), initializer=tf.constant_initializer(alpha_init), trainable=True)

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

class DiffusionNoise(layers.Layer):
    def __init__(self, stddev=0.05, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, x):
        noise = tf.random.normal(tf.shape(x), stddev=self.stddev)
        return x + noise
# =============================================================================
# 3. EEG branches: frequency & channel with diffusion
# =============================================================================
def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = Input(shape=input_shape)
    x = DiffusionNoise(noise_std)(inp)
    x = layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)      # → (batch,64)
    x = DiffusionNoise(noise_std)(x)
    x = DiffuSSMLayer(model_dim=64, hidden_dim=64)(x)
    return models.Model(inp, x, name="FreqBranchDiff")

def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = Input(shape=input_shape)
    x = DiffusionNoise(noise_std)(inp)
    x = layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)      # → (batch,64)
    x = DiffusionNoise(noise_std)(x)
    x = DiffuSSMLayer(model_dim=64, hidden_dim=64)(x)
    return models.Model(inp, x, name="ChanBranchDiff")

# =============================================================================
# 4. rPPG branch: resample + 1D CNN + diffusion
# =============================================================================
def build_rppg_branch(input_length=200, noise_std=0.05):
    inp = Input(shape=(input_length,1))
    x = DiffusionNoise(noise_std)(inp)
    x = layers.Conv1D(16,3,activation='relu',padding='same')(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64,3,activation='relu',padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)      # → (batch,64)
    x = DiffusionNoise(noise_std)(x)
    x = DiffuSSMLayer(model_dim=64, hidden_dim=64)(x)
    return models.Model(inp, x, name="rPPGBranch")

# =============================================================================
# 5. Cross-Modality Gated Attention (EEG ↔ rPPG)
# =============================================================================
class CrossModalGatedAttention(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.Wq = layers.Dense(dim, use_bias=False)
        self.Wk = layers.Dense(dim, use_bias=False)
        self.Wm = layers.Dense(dim)
        self.Wf = layers.Dense(dim)
        self.bf = self.add_weight(
            name="bf",
            shape=(dim,),
            initializer=tf.zeros_initializer(),
            trainable=True
        )
        #self.bf = self.add_weight("bf", shape=(dim,), initializer="zeros")
        self.relu = layers.ReLU()

    def call(self, z_eeg, z_rppg):
        Q = self.Wq(z_rppg)
        K = self.Wk(z_eeg)
        scores = tf.reduce_sum(Q * K, axis=-1, keepdims=True) / tf.math.sqrt(tf.cast(K.shape[-1],tf.float32))
        w = tf.nn.softmax(scores, axis=1)
        A = tf.reduce_sum(w * tf.expand_dims(z_eeg,1), axis=1)  # (batch,dim)

        concat = tf.concat([A, z_rppg], axis=-1)
        f = tf.sigmoid(self.Wf(concat) + self.bf)

        m = self.Wm(A)
        h = self.relu(z_rppg + m * f)
        return h

# =============================================================================
# 6. Data loading & pairing
# =============================================================================
def load_eeg_features(dir_path):
    X,Y,subs,ids = [],[],[],[]
    for fn in os.listdir(dir_path):
        if not fn.endswith(".npy"): continue
        m_lbl = re.search(r'label(\d+)', fn)
        m_sub = re.search(r'subject(\d+)', fn)
        if not m_lbl or not m_sub: continue
        lbl = int(m_lbl.group(1))
        if lbl == 4: continue
        sub = int(m_sub.group(1))
        data = np.load(os.path.join(dir_path, fn))
        T = data.shape[1]; nwin = T//200
        for i in range(nwin):
            X.append(data[:, i*200:(i+1)*200, :])
            Y.append(lbl)
            subs.append(sub)
            ids.append(i)
    return np.array(X), np.array(Y), np.array(subs), np.array(ids)

def load_rppg_features(base_dir, fixed_len=12000, win_len=200):
    X, subs, ids = [], [], []
    for d in sorted(os.listdir(base_dir)):
        sub_dir = os.path.join(base_dir, d)
        if not os.path.isdir(sub_dir): continue
        sub = int(re.sub(r'\D+', '', d)) 
        for fn in os.listdir(sub_dir):
            if not fn.endswith(".npy"): continue
            sig = np.load(os.path.join(sub_dir, fn))  # e.g. (1546,)
            L = len(sig)
            # resample → fixed_len
            xp = np.linspace(0, L-1, fixed_len)
            f  = interp1d(np.arange(L), sig, kind='linear')
            rs = f(xp)
            # window into fixed_len//win_len segments
            nw = fixed_len // win_len
            for i in range(nw):
                w = rs[i*win_len:(i+1)*win_len]
                X.append(w[...,np.newaxis])
                subs.append(sub)
                ids.append(i)
    return np.array(X), np.array(subs), np.array(ids)

# =============================================================================
# 7. Main training: intra-subject loop
# =============================================================================
if __name__ == "__main__":
    EEG_DIR  = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
    RPPG_DIR = "/home/bcml1/sigenv/_4월/_rppg/minipatch_rppg"

    X_eeg, Y, sub_eeg, win_eeg = load_eeg_features(EEG_DIR)
    X_rppg, sub_rppg, win_rppg = load_rppg_features(RPPG_DIR, fixed_len=12000, win_len=200)

    # (subject, window) 을 키로 매핑
    map_r = { (sub_r, w_r): idx for idx, (sub_r, w_r)
              in enumerate(zip(sub_rppg, win_rppg)) }

    pairs = []
    for idx_e, (sub_e, w_e) in enumerate(zip(sub_eeg, win_eeg)):
        key = (sub_e, w_e)
        if key in map_r:
            pairs.append((idx_e, map_r[key]))

    if not pairs:
        raise ValueError("페어링된 샘플이 없습니다. 경로와 naming을 다시 확인하세요.")

    idx_e, idx_r = zip(*pairs)
    X_eeg = X_eeg[list(idx_e)]
    X_rppg = X_rppg[list(idx_r)]
    Y      = Y[list(idx_e)]
    subs   = sub_eeg[list(idx_e)]
    print(f"🔄 Paired {len(Y)} windows across EEG & rPPG")

    # prepare static branches & fusion layer
    freq_branch = build_freq_branch_with_diffusion()
    chan_branch = build_chan_branch_with_diffusion()
    rppg_branch = build_rppg_branch()
    cmga_layer  = CrossModalGatedAttention(dim=64)

    for s in np.unique(subs):
        mask = (subs == s)
        Xe, Xr, y = X_eeg[mask], X_rppg[mask], Y[mask]
        if len(y) < 20: continue

        # remap labels
        ul = np.unique(y); lm = {v:i for i,v in enumerate(ul)}
        y = np.array([lm[v] for v in y]); nc = len(ul)

        # split
        Xe_tr, Xe_tmp, Xr_tr, Xr_tmp, y_tr, y_tmp = train_test_split(
            Xe, Xr, y, test_size=0.3, random_state=42, stratify=y)
        Xe_val, Xe_te, Xr_val, Xr_te, y_val, y_te = train_test_split(
            Xe_tmp, Xr_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

        # balance by augmentation
        counts = np.bincount(y_tr); mx = counts.max()
        aug_e, aug_r, aug_y = [],[],[]
        for lbl,cnt in enumerate(counts):
            idxs = np.where(y_tr==lbl)[0]
            for _ in range(mx-cnt):
                i = np.random.choice(idxs)
                aug_e.append(add_diffusion_noise(Xe_tr[i],0.05).numpy())
                aug_r.append(add_diffusion_noise(Xr_tr[i],0.05).numpy())
                aug_y.append(lbl)
        if aug_e:
            Xe_tr = np.concatenate([Xe_tr, np.array(aug_e)],axis=0)
            Xr_tr = np.concatenate([Xr_tr, np.array(aug_r)],axis=0)
            y_tr  = np.concatenate([y_tr,  np.array(aug_y)],axis=0)

        train_ds = tf.data.Dataset.from_tensor_slices(((Xe_tr,Xr_tr),y_tr))\
                         .shuffle(1000).batch(16)
        val_ds   = tf.data.Dataset.from_tensor_slices(((Xe_val,Xr_val),y_val)).batch(16)
        test_ds  = tf.data.Dataset.from_tensor_slices(((Xe_te,Xr_te),y_te)).batch(16)

        # build model
        in_eeg = Input(shape=(4,200,8), name="EEG_In")
        in_r  = Input(shape=(200,1), name="rPPG_In")

        # EEG branches
        feq = freq_branch(layers.Permute((2,3,1))(in_eeg))
        fch = chan_branch(layers.Permute((2,1,3))(in_eeg))
        fe  = layers.Concatenate()([feq, fch])        # (batch,128)
        fe  = layers.Dense(64, activation='relu')(fe) # project to 64

        fr  = rppg_branch(in_r)                       # (batch,64)
        fused = cmga_layer(fe, fr)                    # (batch,64)

        x = layers.Dense(128,activation='relu')(fused)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(nc, activation='softmax')(x)

        model = models.Model([in_eeg,in_r], out, name=f"Subj_{s}_MM")
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print(f"\n▶ Training Subject {s}, labels={ul}")
        hist = model.fit(train_ds, epochs=300,
                         validation_data=val_ds,
                         callbacks=[EarlyStopping('val_accuracy',patience=50,restore_best_weights=True)],
                         verbose=1)

        # save curves & metrics
        sd = os.path.join(RESULT_DIR, f"subj{s}")
        os.makedirs(sd, exist_ok=True)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(hist.history['loss'], label='train loss')
        plt.plot(hist.history['val_loss'], label='val loss')
        plt.legend(); plt.title('Loss')
        plt.subplot(1,2,2)
        plt.plot(hist.history['accuracy'], label='train acc')
        plt.plot(hist.history['val_accuracy'], label='val acc')
        plt.legend(); plt.title('Accuracy')
        plt.savefig(os.path.join(sd,"curves.png")); plt.close()

        test_l, test_a = model.evaluate(test_ds, verbose=0)
        print(f"✅ Subject {s} Test: loss={test_l:.4f}, acc={test_a:.4f}")
        ypred = np.argmax(model.predict(test_ds),axis=1)
        cm = confusion_matrix(y_te, ypred)
        report = classification_report(y_te, ypred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title("Confusion Matrix"); plt.savefig(os.path.join(sd,"cm.png")); plt.close()
        with open(os.path.join(sd,"report.txt"),"w") as f:
            f.write(report)
        model.save(os.path.join(sd,"model.keras"))

    print("🎯 All subjects complete.")
