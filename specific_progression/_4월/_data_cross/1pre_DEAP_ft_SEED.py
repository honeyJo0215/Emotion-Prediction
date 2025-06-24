import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================
DEAP_DIR       = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
SEED_DIR       = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"
CROSS_RESULT   = "/home/bcml1/sigenv/_4월/_data_cross/A_1pre_DEAP_ft_SEED"
PRETRAIN_DIR   = os.path.join(CROSS_RESULT, "pretrain")
os.makedirs(PRETRAIN_DIR, exist_ok=True)

# =============================================================================
# GPU memory limiter
# =============================================================================
def limit_gpu_memory(memory_limit_mib=10000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
            )
            print(f"GPU memory limited to {memory_limit_mib} MiB.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU.")
limit_gpu_memory(10000)

# =============================================================================
# DiffuSSMLayer: small denoiser block
# =============================================================================
class DiffuSSMLayer(layers.Layer):
    def __init__(self, hidden_dim=64, output_units=None, **kwargs):
        super().__init__(**kwargs)
        self.dense1    = layers.Dense(hidden_dim, activation='relu')
        self.norm1     = layers.LayerNormalization()
        self.dense2    = layers.Dense(hidden_dim, activation='relu')
        self.norm2     = layers.LayerNormalization()
        final_units   = output_units if output_units is not None else hidden_dim
        self.out_dense= layers.Dense(final_units, activation=None)
        self.norm_out  = layers.LayerNormalization()

    def call(self, x, training=False):
        h = self.dense1(x)
        h = self.norm1(h)
        h = self.dense2(h)
        h = self.norm2(h)
        out = self.out_dense(h)
        out = self.norm_out(out)
        return out

# =============================================================================
# Add Gaussian noise
# =============================================================================
def add_diffusion_noise(x, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)
    return x + noise

# =============================================================================
# Build CNN branches with diffusion pre- and post- noise
# =============================================================================
def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x   = layers.Lambda(lambda t: add_diffusion_noise(t, stddev=noise_std))(inp)
    x   = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x   = layers.GlobalAveragePooling2D()(x)
    noisy = layers.Lambda(lambda t: add_diffusion_noise(t, stddev=noise_std))(x)
    res   = DiffuSSMLayer(hidden_dim=64)(noisy)
    out   = layers.Add()([x, res])
    return models.Model(inp, out, name="FreqBranchDiff")

def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x   = layers.Lambda(lambda t: add_diffusion_noise(t, stddev=noise_std))(inp)
    x   = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x   = layers.GlobalAveragePooling2D()(x)
    noisy = layers.Lambda(lambda t: add_diffusion_noise(t, stddev=noise_std))(x)
    res   = DiffuSSMLayer(hidden_dim=64)(noisy)
    out   = layers.Add()([x, res])
    return models.Model(inp, out, name="ChanBranchDiff")

def build_separated_model_with_diffusion(num_classes=4, noise_std=0.05):
    inp = layers.Input(shape=(4,200,8), name="CSP_Input")
    # Frequency path
    f_in = layers.Permute((2,3,1))(inp)  # -> (200,8,4)
    f_b  = build_freq_branch_with_diffusion((200,8,4), noise_std)
    f_out= f_b(f_in)
    # Channel path
    c_in = layers.Permute((2,1,3))(inp)  # -> (200,4,8)
    c_b  = build_chan_branch_with_diffusion((200,4,8), noise_std)
    c_out= c_b(c_in)
    # Classifier
    x    = layers.Concatenate()([f_out, c_out])
    x    = layers.Dense(128, activation='relu')(x)
    x    = layers.Dropout(0.3)(x)
    x    = layers.Dense(64, activation='relu')(x)
    x    = layers.Dropout(0.3)(x)
    out  = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out, name="Separated_CNN_Diffusion")

# =============================================================================
# Load DEAP CSP features
# =============================================================================
# =============================================================================
# DEAP CSP 특징 불러오기 (128Hz → 200Hz로 업샘플링)
# =============================================================================
def load_deap_csp_features(dir_path):
    X_list, Y_list = [], []
    for fn in os.listdir(dir_path):
        if not fn.endswith(".npy"):
            continue
        # data shape: (4, T, 8)
        data = np.load(os.path.join(dir_path, fn))
        # 파일 이름에서 레이블(label0~3) 추출
        label_m = re.search(r'label(\d+)', fn, re.IGNORECASE)
        if not label_m:
            continue
        label = int(label_m.group(1))
        if label not in [0,1,2,3]:
            continue

        # ① 원본 128Hz 데이터를 1초(128샘플) 단위로 나누기
        orig_T = data.shape[1]
        sec_count = orig_T // 128
        for i_sec in range(sec_count):
            segment = data[:, i_sec*128:(i_sec+1)*128, :]  # (4,128,8)

            # ② 선형 보간을 이용해 128 → 200 샘플로 업샘플링
            new_T = 200
            orig_idx = np.arange(128)
            new_idx  = np.linspace(0, 127, new_T)
            resampled = np.zeros((segment.shape[0], new_T, segment.shape[2]),
                                 dtype=segment.dtype)
            for ch in range(segment.shape[0]):
                for feat in range(segment.shape[2]):
                    resampled[ch, :, feat] = np.interp(
                        new_idx,      # 보간 지점 (0~127 사이 200개)
                        orig_idx,     # 원본 인덱스 (0~127)
                        segment[ch, :, feat]  # 원본 신호
                    )

            # ③ 업샘플링된 1초 구간을 리스트에 추가
            X_list.append(resampled)  # (4,200,8)
            Y_list.append(label)

    X = np.array(X_list)  # (n_samples, 4, 200, 8)
    Y = np.array(Y_list)
    print(f"[DEAP] 총 {X.shape[0]}개 샘플 로드 (200Hz로 리샘플링 완료)")
    return X, Y


# =============================================================================
# Load SEED CSP features
# =============================================================================
def load_seed_csp_features(dir_path):
    X_list, Y_list, F_list = [], [], []
    for fn in os.listdir(dir_path):
        if not fn.endswith(".npy"): continue
        data = np.load(os.path.join(dir_path, fn))
        T = data.shape[1]
        n_w = T // 200
        label_m   = re.search(r'label(\d+)', fn, re.IGNORECASE)
        subject_m = re.search(r'subject(\d+)', fn, re.IGNORECASE)
        if not label_m or not subject_m: continue
        label  = int(label_m.group(1))
        subj   = subject_m.group(1)
        for i in range(n_w):
            X_list.append(data[:, i*200:(i+1)*200, :])
            Y_list.append(label)
            F_list.append(f"{fn}_win{i}")
    X = np.array(X_list)
    Y = np.array(Y_list)
    F = np.array(F_list)
    print(f"[SEED] Loaded {X.shape[0]} samples")
    return X, Y, F

# =============================================================================
# Pre-train on DEAP
# =============================================================================
X_deap, Y_deap = load_deap_csp_features(DEAP_DIR)
# 80% train, 10% val, 10% test
X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(X_deap, Y_deap,
                                             test_size=0.2,
                                             stratify=Y_deap,
                                             random_state=42)
X_val, X_te, Y_val, Y_te = train_test_split(X_tmp, Y_tmp,
                                            test_size=0.5,
                                            stratify=Y_tmp,
                                            random_state=42)

ds_tr  = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr)).shuffle(2000).batch(16)
ds_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(16)
ds_te  = tf.data.Dataset.from_tensor_slices((X_te, Y_te)).batch(16)

model = build_separated_model_with_diffusion(num_classes=4, noise_std=0.02)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
es = EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)
history = model.fit(ds_tr, epochs=300, validation_data=ds_val, callbacks=[es])

# save pretrain model weights
pretrain_w = os.path.join(PRETRAIN_DIR, "pretrained.weights.h5")
model.save_weights(pretrain_w)

# plot & save curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("DEAP Pretrain Loss")
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title("DEAP Pretrain Acc")
plt.savefig(os.path.join(PRETRAIN_DIR, "pretrain_curves.png"))
plt.close()

# evaluate on DEAP test
te_loss, te_acc = model.evaluate(ds_te, verbose=0)
y_pred = np.argmax(model.predict(ds_te), axis=1)
cm = confusion_matrix(Y_te, y_pred)
rep = classification_report(Y_te, y_pred)
# save reports
with open(os.path.join(PRETRAIN_DIR, "deap_test_report.txt"), "w") as f:
    f.write(f"Loss: {te_loss:.4f}, Acc: {te_acc:.4f}\n\n")
    f.write(rep)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("DEAP Test Confusion")
plt.savefig(os.path.join(PRETRAIN_DIR, "deap_confusion.png"))
plt.close()


# =============================================================================
# Fine-tune & test on SEED, per subject
# =============================================================================
X_seed, Y_seed, F_seed = load_seed_csp_features(SEED_DIR)
subjects = sorted(set(int(re.search(r'subject(\d+)', f).group(1)) for f in F_seed))

for subj in subjects:
    # only subjects 2–9
    if not (2 <= subj < 10):
        continue
    s_mask = np.array([f"subject{subj}" in fname for fname in F_seed])
    X_s, Y_s, F_s = X_seed[s_mask], Y_seed[s_mask], F_seed[s_mask]

    # file-level split: train/val/test
    files = np.unique(F_s)
    tr_f, te_f = train_test_split(files, test_size=0.3, random_state=42)
    tr_f, val_f = train_test_split(tr_f, test_size=0.25, random_state=42)

    mask_tr  = np.isin(F_s, tr_f)
    mask_val = np.isin(F_s, val_f)
    mask_te  = np.isin(F_s, te_f)

    X_tr_s, Y_tr_s = X_s[mask_tr], Y_s[mask_tr]
    X_val_s, Y_val_s = X_s[mask_val], Y_s[mask_val]
    X_te_s, Y_te_s = X_s[mask_te], Y_s[mask_te]

    subj_dir = os.path.join(CROSS_RESULT, f"s{subj}")
    os.makedirs(subj_dir, exist_ok=True)

    ds_tr_s  = tf.data.Dataset.from_tensor_slices((X_tr_s, Y_tr_s)).shuffle(500).batch(16)
    ds_val_s = tf.data.Dataset.from_tensor_slices((X_val_s, Y_val_s)).batch(16)
    ds_te_s  = tf.data.Dataset.from_tensor_slices((X_te_s, Y_te_s)).batch(16)

    # build & load pretrain weights
    ft_model = build_separated_model_with_diffusion(num_classes=len(np.unique(Y_s)), noise_std=0.02)
    ft_model.load_weights(pretrain_w)
    ft_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                     loss="sparse_categorical_crossentropy",
                     metrics=["accuracy"])
    es_s = EarlyStopping(monitor='val_accuracy', patience=500, restore_best_weights=True)
    history_s = ft_model.fit(ds_tr_s, epochs=1000, validation_data=ds_val_s, callbacks=[es_s])

    # save fine-tuned model
    ft_model.save(os.path.join(subj_dir, "model_finetuned.keras"))

    # curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history_s.history['loss'], label='Train Loss')
    plt.plot(history_s.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title(f"Subject {subj} Loss")
    plt.subplot(1,2,2)
    plt.plot(history_s.history['accuracy'], label='Train Acc')
    plt.plot(history_s.history['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title(f"Subject {subj} Acc")
    plt.savefig(os.path.join(subj_dir, "finetune_curves.png"))
    plt.close()

    # evaluate & report
    loss_s, acc_s = ft_model.evaluate(ds_te_s, verbose=0)
    y_pred_s = np.argmax(ft_model.predict(ds_te_s), axis=1)
    cm_s = confusion_matrix(Y_te_s, y_pred_s)
    rep_s = classification_report(Y_te_s, y_pred_s)
    with open(os.path.join(subj_dir, "seed_test_report.txt"), "w") as f:
        f.write(f"Loss: {loss_s:.4f}, Acc: {acc_s:.4f}\n\n")
        f.write(rep_s)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_s, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Subject {subj} Confusion")
    plt.savefig(os.path.join(subj_dir, "seed_confusion.png"))
    plt.close()

print("Cross-dataset pretrain & fine-tune complete.")
