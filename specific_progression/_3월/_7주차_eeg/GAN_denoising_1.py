import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, Dense, LayerNormalization, Dropout, 
                                     Flatten, Input, Lambda, Concatenate, TimeDistributed, 
                                     GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 결과 저장 경로 설정
RESULT_DIR = "/home/bcml1/sigenv/_7주차_eeg/Diffu_GAN_1"
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# GPU 메모리 제한
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
limit_gpu_memory(9000)

# =============================================================================
# 전처리 Layer
class PreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        x = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
        axis = list(range(1, inputs.shape.rank))
        min_val = tf.reduce_min(x, axis=axis, keepdims=True)
        max_val = tf.reduce_max(x, axis=axis, keepdims=True)
        range_val = max_val - min_val + 1e-8
        return (x - min_val) / range_val

# =============================================================================
# 2D mapping 함수 (각 채널의 4밴드 데이터를 8x8 이미지로 매핑)
def map_channels_to_2d(segment):
    num_channels, num_bands = segment.shape
    mapped = np.zeros((8, 8, num_bands))
    for band in range(num_bands):
        vec = segment[:, band]
        vec_padded = np.pad(vec, (0, 64 - num_channels), mode='constant')
        mapped[:, :, band] = vec_padded.reshape(8, 8)
    return mapped

# =============================================================================
# subject normalize
def subject_normalize(X, S):
    X_norm = np.copy(X)
    unique_subjects = np.unique(S)
    for subj in unique_subjects:
        subj_idx = np.where(S == subj)[0]
        de_data = X[subj_idx, :, 0, ...]
        mean_de = np.mean(de_data)
        X_norm[subj_idx, :, 0, ...] = X_norm[subj_idx, :, 0, ...] - mean_de
        psd_data = X[subj_idx, :, 1, ...]
        mean_psd = np.mean(psd_data)
        X_norm[subj_idx, :, 1, ...] = X_norm[subj_idx, :, 1, ...] - mean_psd
    return X_norm

# =============================================================================
# 데이터 로드 함수 (파일 ID도 함께 저장)
def load_all_data(de_base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE",
                  psd_base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_PSD",
                  window_size=10, step_size=1):
    X, Y, S, F = [], [], [], []
    for subfolder in ["1", "2", "3"]:
        de_folder = os.path.join(de_base_dir, subfolder)
        psd_folder = os.path.join(psd_base_dir, subfolder)
        if not os.path.isdir(de_folder) or not os.path.isdir(psd_folder):
            continue
        for file_name in os.listdir(de_folder):
            if not file_name.endswith(".npy"):
                continue
            de_file_path = os.path.join(de_folder, file_name)
            psd_file_path = os.path.join(psd_folder, file_name)
            try:
                de_data = np.load(de_file_path)   # (62, n_seconds, 4)
                psd_data = np.load(psd_file_path)  # (62, n_seconds, 4)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue
            try:
                label = int(file_name.split('_')[-1].split('.')[0])
            except:
                print(f"Label extraction failed for {file_name}")
                continue
            subject_id = file_name.split('_')[0]
            n_seconds = de_data.shape[1]
            segments = []
            for t in range(n_seconds):
                de_segment = de_data[:, t, :]
                psd_segment = psd_data[:, t, :]
                de_mapped = map_channels_to_2d(de_segment)
                psd_mapped = map_channels_to_2d(psd_segment)
                de_img = np.transpose(de_mapped, (2, 0, 1))
                de_img = np.expand_dims(de_img, axis=-1)
                psd_img = np.transpose(psd_mapped, (2, 0, 1))
                psd_img = np.expand_dims(psd_img, axis=-1)
                sample = np.stack([de_img, psd_img], axis=0)  # (2, 4, 8, 8, 1)
                segments.append(sample)
            if len(segments) >= window_size:
                for i in range(0, len(segments) - window_size + 1, step_size):
                    window = np.stack(segments[i:i+window_size], axis=0)  # (10, 2, 4, 8, 8, 1)
                    X.append(window)
                    Y.append(label)
                    S.append(subject_id)
                    F.append(file_name)
            print(f"Loaded {n_seconds} seconds from {file_name} with label {label} and subject {subject_id}.")
    X = np.array(X)
    Y = np.array(Y)
    S = np.array(S)
    F = np.array(F)
    print(f"Total samples (windows): {X.shape[0]}, each sample shape: {X.shape[1:]}")
    return X, Y, S, F

# 전체 데이터 로드 및 normalize
X, Y, subjects, file_ids = load_all_data(window_size=10, step_size=1)
X = subject_normalize(X, subjects)

# =============================================================================
# Dynamic Graph Convolution Layer (Dense 층은 __init__에서 생성)
class DynamicGraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(DynamicGraphConvLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.dense_proj = Dense(output_dim, activation='relu', kernel_regularizer=l2(1e-5))
        self.fc1 = Dense(output_dim, activation='elu', kernel_regularizer=l2(1e-5))
        self.fc2 = Dense(output_dim, activation='tanh', kernel_regularizer=l2(1e-5))
    def call(self, inputs):
        proj = self.dense_proj(inputs)
        A = tf.matmul(proj, proj, transpose_b=True)
        A = tf.nn.tanh(A)
        A = tf.nn.relu(A)
        H = tf.matmul(A, inputs)
        H = self.fc1(H)
        H = self.fc2(H)
        return H

# =============================================================================
# Adapter Module for Transfer Learning
class AdapterModule(tf.keras.layers.Layer):
    def __init__(self, d_model, bottleneck_dim=16, **kwargs):
        super(AdapterModule, self).__init__(**kwargs)
        self.down = Dense(bottleneck_dim, activation='elu', kernel_regularizer=l2(1e-5))
        self.up = Dense(d_model, activation=None, kernel_regularizer=l2(1e-5))
    def call(self, x):
        shortcut = x
        x = self.down(x)
        x = self.up(x)
        return x + shortcut

# =============================================================================
# Dual-Branch Single-Segment Model (DE와 PSD 각각 분리 후 Dynamic GCN 적용)
def build_single_segment_model(d_model=64, num_nodes=62):
    input_seg = Input(shape=(2, 4, 8, 8, 1))
    de_input = Lambda(lambda x: x[:, 0, ...])(input_seg)
    psd_input = Lambda(lambda x: x[:, 1, ...])(input_seg)
    flat_de = Flatten()(de_input)
    flat_psd = Flatten()(psd_input)
    x_de = Dense(num_nodes * d_model, activation="relu", kernel_regularizer=l2(1e-5))(flat_de)
    x_de = Lambda(lambda x: tf.reshape(x, (-1, num_nodes, d_model)))(x_de)
    x_psd = Dense(num_nodes * d_model, activation="relu", kernel_regularizer=l2(1e-5))(flat_psd)
    x_psd = Lambda(lambda x: tf.reshape(x, (-1, num_nodes, d_model)))(x_psd)
    H_de = DynamicGraphConvLayer(d_model)(x_de)
    H_psd = DynamicGraphConvLayer(d_model)(x_psd)
    H_fused = Concatenate(axis=-1)([H_de, H_psd])
    fused = Dense(d_model, activation="relu", kernel_regularizer=l2(1e-5))(H_fused)
    out_feature = Lambda(lambda x: tf.reduce_mean(x, axis=1))(fused)
    model = Model(inputs=input_seg, outputs=out_feature)
    return model

# =============================================================================
# Time Embedding 레이어 (sinusoidal 방식)
class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(TimeEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
    def call(self, t):
        # t: (batch,) 또는 (batch,1)
        half_dim = self.d_model // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        t = tf.cast(tf.reshape(t, [-1, 1]), tf.float32)
        emb = t * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb  # shape: (batch, d_model)

# =============================================================================
# 수정된 DiffuSSM 레이어 (time embedding을 받아 노이즈 예측에 반영)
class DiffuSSMLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.5, bottleneck_dim=16, **kwargs):
        super(DiffuSSMLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.time_dense = Dense(d_model, activation=tf.nn.gelu, kernel_regularizer=l2(1e-5))
        # Hourglass block
        self.hourglass_down = Dense(d_model // 2, activation=tf.nn.gelu, kernel_regularizer=l2(1e-5))
        self.hourglass_up = Dense(d_model, activation=None, kernel_regularizer=l2(1e-5))
        # Gated bidirectional SSM (Bidirectional GRU)
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(d_model, return_sequences=True), merge_mode='sum'
        )
        self.dropout = Dropout(dropout_rate)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        # Feed Forward Network (FFN)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation=tf.nn.gelu, kernel_regularizer=l2(1e-5)),
            Dense(d_model, kernel_regularizer=l2(1e-5))
        ])
        self.dropout_ffn = Dropout(dropout_rate)
        self.layernorm_ffn = LayerNormalization(epsilon=1e-6)
        # Adapter Module
        self.adapter = AdapterModule(d_model, bottleneck_dim)
        # AdaLN-Zero: gating 파라미터
        self.alpha = self.add_weight(shape=(1,), initializer=tf.zeros_initializer(), trainable=True, name="ssm_alpha")
    
    def call(self, inputs, time_emb, training=False):
        # time_emb: (batch, d_model) -> expand to (batch, 1, d_model) and broadcast
        time_emb_exp = tf.expand_dims(time_emb, axis=1)
        x = inputs + time_emb_exp
        # Hourglass branch
        x_hour = self.hourglass_down(x)
        x_hour = self.hourglass_up(x_hour)
        x_hour = self.dropout(x_hour, training=training)
        # SSM branch (Bidirectional GRU)
        ssm_out = self.bi_gru(x, training=training)
        ssm_out = self.dropout(ssm_out, training=training)
        ssm_out = self.alpha * ssm_out
        # Skip connection
        x = x + x_hour + ssm_out
        x = self.layernorm(x)
        # FFN
        ffn_out = self.ffn(x)
        ffn_out = self.dropout_ffn(ffn_out, training=training)
        x = self.layernorm_ffn(x + ffn_out)
        # Adapter Module
        x = self.adapter(x)
        return x

# =============================================================================
# GAN 기반 Denoiser Layer (Generator 역할)
# 수정: 입력 shape (batch, time, features)에 맞게 Dense 계층으로 처리하도록 변경
class GANDenoiser(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(GANDenoiser, self).__init__(**kwargs)
        self.dense1 = Dense(d_model, activation='relu', kernel_regularizer=l2(1e-5))
        self.dense2 = Dense(d_model, activation='linear', kernel_regularizer=l2(1e-5))
    def call(self, x):
        # x: (batch, time, features)
        residual = x
        x = self.dense1(x)
        x = self.dense2(x)
        return residual + x

# =============================================================================
# 기존 ReverseDiffusionLayer는 GAN 기반 denoising으로 대체됨
# 최종 모델: TimeDistributed CNN으로 feature 추출 후, 
# GAN 기반 denoiser로 노이즈 제거하고 감정 분류 수행
def build_model(input_shape=(10, 2, 4, 8, 8, 1), n_diffusion=10, d_ff=512, p_drop=0.5, d_model=64, noise_std=0.1):
    inputs = Input(shape=input_shape)
    # Feature extraction
    single_seg_model = build_single_segment_model(d_model)
    features_seq = TimeDistributed(single_seg_model, name="TimeDistributed_CNN")(inputs)
    
    # Diffusion forward: 노이즈 추가
    def add_diffusion_noise(x):
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_std)
        return x + noise
    noisy_features_seq = Lambda(add_diffusion_noise, name="add_diffusion_noise")(features_seq)
    
    # GAN-based denoising: 기존 역확산 과정 대신 GAN denoiser 사용
    x = GANDenoiser(d_model)(noisy_features_seq)
    
    # 분류: Flatten 후 softmax 분류기
    x_flat = Flatten()(x)
    outputs = Dense(4, activation="softmax", kernel_regularizer=l2(1e-5))(x_flat)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# =============================================================================
# Fine tuning 분리 함수 (파일별, 중간 20% 구간 분리)
def split_finetune_test_by_file(X_test, Y_test, F_test, ft_ratio=0.2):
    X_ft_list, X_test_final_list = [], []
    y_ft_list, y_test_final_list = [], []
    unique_files = np.unique(F_test)
    for f in unique_files:
        indices = np.where(F_test == f)[0]
        n = len(indices)
        if n < 2:
            continue
        start = int(n * 0.4)
        end = int(n * 0.6)
        if end <= start:
            ft_idx = indices[-1:]
            test_idx = indices[:-1]
        else:
            ft_idx = indices[start:end]
            test_idx = np.concatenate([indices[:start], indices[end:]])
        if len(ft_idx) > 0 and len(test_idx) > 0:
            X_ft_list.append(X_test[ft_idx])
            y_ft_list.append(Y_test[ft_idx])
            X_test_final_list.append(X_test[test_idx])
            y_test_final_list.append(Y_test[test_idx])
    if len(X_ft_list) == 0:
        X_shape = (0,) + X_test.shape[1:]
        return np.empty(X_shape), np.empty(X_shape), np.empty((0,)), np.empty((0,))
    X_ft = np.concatenate(X_ft_list, axis=0)
    y_ft = np.concatenate(y_ft_list, axis=0)
    X_test_final = np.concatenate(X_test_final_list, axis=0)
    y_test_final = np.concatenate(y_test_final_list, axis=0)
    return X_ft, X_test_final, y_ft, y_test_final

# =============================================================================
# LOSO 방식: 각 subject을 leave-one-out 방식으로 test로 사용
unique_subjects = np.unique(subjects)
X_selected = X
Y_selected = Y
F_selected = np.array(file_ids)
S_selected = subjects

for subj in unique_subjects:
    if not (1 <= int(subj) < 15):
        continue
    print(f"\n========== LOSO: Test subject = {subj} ==========")
    test_mask = (S_selected == subj)
    train_mask = (S_selected != subj)
    
    X_test = X_selected[test_mask]
    Y_test = Y_selected[test_mask]
    F_test = F_selected[test_mask]
    X_train_valid = X_selected[train_mask]
    Y_train_valid = Y_selected[train_mask]
    
    if X_train_valid.shape[0] == 0:
        print(f"Warning: Train/Valid set is empty for subject {subj}. Skipping...")
        continue
    
    # train/valid split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_valid, Y_train_valid, test_size=0.25, random_state=42, stratify=Y_train_valid
    )
    
    # 파일별로 fine tuning과 test 데이터 분리
    X_ft, X_test_final, y_ft, y_test_final = split_finetune_test_by_file(X_test, Y_test, F_test, ft_ratio=0.2)
    
    subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
    os.makedirs(subj_folder, exist_ok=True)
    
    with tf.device('/CPU:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(64)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
    
    model = build_model(input_shape=(10, 2, 4, 8, 8, 1))
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4, decay_steps=100000, decay_rate=0.9, staircase=True))
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, mode='max', restore_best_weights=True)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    
    history = model.fit(train_dataset, epochs=150, validation_data=val_dataset, callbacks=[early_stopping])
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    train_curve_path = os.path.join(subj_folder, "training_curves.png")
    plt.savefig(train_curve_path)
    plt.close()
    print(f"Subject {subj}: Training curves saved to {train_curve_path}")
    
    # Fine tuning 단계
    ft_dataset = tf.data.Dataset.from_tensor_slices((X_ft, y_ft)).shuffle(1000).batch(64)
    print(f"Subject {subj}: Fine tuning with {X_ft.shape[0]} samples.")
    model.fit(ft_dataset, epochs=20)
    
    test_dataset_final = tf.data.Dataset.from_tensor_slices((X_test_final, y_test_final)).batch(64)
    test_loss, test_acc = model.evaluate(test_dataset_final)
    print(f"Subject {subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    y_pred_prob = model.predict(test_dataset_final)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    cm = confusion_matrix(y_test_final, y_pred)
    report = classification_report(y_test_final, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(subj_folder, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Subject {subj}: Confusion matrix saved to {cm_path}")
    
    report_path = os.path.join(subj_folder, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Subject {subj}: Classification report saved to {report_path}")
    
    model_save_path = os.path.join(subj_folder, "model_eeg_cnn_diffussm.keras")
    model.save(model_save_path)
    print(f"Subject {subj}: Model saved to {model_save_path}")
