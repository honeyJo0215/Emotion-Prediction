import os
import re
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# GPU 메모리 제한
def limit_gpu_memory(memory_limit_mib=5000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)])
        except RuntimeError as e:
            print(e)
limit_gpu_memory()

# DiffuSSM Layer
class DiffuSSMLayer(layers.Layer):
    def __init__(self, hidden_dim=64, output_units=None, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.norm1 = layers.LayerNormalization()
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.norm2 = layers.LayerNormalization()
        final_units = output_units if output_units else hidden_dim
        self.out_dense = layers.Dense(final_units)
        self.norm_out = layers.LayerNormalization()

    def call(self, x):
        h = self.norm1(self.dense1(x))
        h = self.norm2(self.dense2(h))
        return self.norm_out(self.out_dense(h))

# Noise addition
def add_diffusion_noise(x, stddev=0.05):
    return x + tf.random.normal(tf.shape(x), stddev=stddev)

# Frequency branch
def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    noisy = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(x)
    diff = DiffuSSMLayer()(noisy)
    out = layers.Add()([x, diff])
    return models.Model(inp, out)

# Channel branch
def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    noisy = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(x)
    diff = DiffuSSMLayer()(noisy)
    out = layers.Add()([x, diff])
    return models.Model(inp, out)

# Full model
def build_separated_model_with_diffusion(num_classes=4, noise_std=0.05):
    inp = layers.Input(shape=(4,200,8), dtype=tf.float32, name='CSP_Input')
    f = layers.Permute((2,3,1))(inp)
    freq_branch = build_freq_branch_with_diffusion(noise_std=noise_std)
    f_feat = freq_branch(f)
    c = layers.Permute((2,1,3))(inp)
    chan_branch = build_chan_branch_with_diffusion(noise_std=noise_std)
    c_feat = chan_branch(c)
    x = layers.Concatenate()([f_feat, c_feat])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inp, outputs=out, name='Separated_CNN_Model_with_Diffusion')

# DEAP 데이터 로딩 함수
def load_deap_csp_features(dir):
    X, Y, subjects = [], [], []
    for f in os.listdir(dir):
        if f.endswith('.npy'):
            data = np.load(os.path.join(dir, f))
            label = int(re.search(r'label(\d+)', f).group(1))
            subj = int(re.search(r'subject(\d+)', f).group(1))
            T = data.shape[1]
            for i in range(T//200):
                X.append(data[:, i*200:(i+1)*200, :])
                Y.append(label)
                subjects.append(subj)
    return np.array(X), np.array(Y), np.array(subjects)

# SEED 데이터 로딩 함수
def load_seed_csp_features(dir):
    return load_deap_csp_features(dir)

experiments = [
    ('DEAP', 'SEED', range(1,16), load_deap_csp_features, load_seed_csp_features),
    ('SEED', 'DEAP', range(1,33), load_seed_csp_features, load_deap_csp_features)
]

for src, tgt, subs, loader_src, loader_tgt in experiments:
    if(src == 'DEAP'):
        Xs, ys, _ = loader_src("/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP")
    else:
        Xs, ys, _ = loader_src("/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP")
    
    xs, Ys = Xs.astype('float32'), ys.astype('int32')

    X_train_pre, X_val_pre, y_train_pre, y_val_pre = train_test_split(
        xs, Ys, test_size=0.2, stratify=Ys, random_state=42)
    
    X_train_pre, X_val_pre, y_train_pre, y_val_pre = X_train_pre.astype('float32'), X_val_pre.astype('float32'), y_train_pre.astype('int32'), y_val_pre.astype('int32')
    
    print(f"X_train_pre.dtype, X_train_pre.shape: {X_train_pre.dtype, X_train_pre.shape}")  # 기대: float32
    print(f"y_train_pre.dtype, y_train_pre.shape: {y_train_pre.dtype, y_train_pre.shape}")  # 기대: int32
    
    model = build_separated_model_with_diffusion()
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    model.fit(X_train_pre, y_train_pre, validation_data=(X_val_pre, y_val_pre), epochs=20, batch_size=32)
    w_pre = model.get_weights()

    if(tgt == 'DEAP'):
        Xt, yt, sub_ids = loader_tgt("/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP")
    else:
        Xt, yt, sub_ids = loader_tgt("/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP")
    Xt, yt, sub_ids = Xt.astype('float32'), yt.astype('int32'), sub_ids.astype(int)

    y_true, y_pred = [], []

    for sub in subs:
        tf.keras.backend.clear_session(); gc.collect()

        mask = (sub_ids == sub)
        Xsub, ysub = Xt[mask], yt[mask]

        X_tv, X_te, y_tv, y_te = train_test_split(Xsub, ysub, test_size=0.3, stratify=ysub, random_state=42)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tv, y_tv, test_size=1/7, stratify=y_tv, random_state=42)
        print(f"X_tv.dtype, X_te.shape: {X_tv.dtype, X_te.shape}")  # 기대: float32
        print(f"y_tv.dtype, y_te.shape: {y_tv.dtype, y_te.shape}")  # 기대: int32
        
        print(f"X_tr.dtype, X_val.shape:{X_tr.dtype, X_val.shape}")  # 기대: float32
        print(f"y_tr.dtype, y_val.shape:{y_tr.dtype, y_val.shape}")  # 기대: int32
    
        ft_model = build_separated_model_with_diffusion()
        ft_model.set_weights(w_pre)
        ft_model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
        ft_model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=10, batch_size=16, verbose=0)

        preds = ft_model.predict(X_te).argmax(axis=1)
        y_true.extend(y_te); y_pred.extend(preds)

    print(classification_report(y_true, y_pred))

    # 결과 저장 (이 부분도 들여쓰기 수정: 피실험자 loop 밖으로 빼내기)
    with open(f"{loader_tgt}_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{loader_tgt} Confusion Matrix")
    plt.savefig(f"{loader_tgt}_cm.png")
    plt.clf()

    print(f"{loader_tgt} done")