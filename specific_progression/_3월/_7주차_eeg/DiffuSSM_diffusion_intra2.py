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
RESULT_DIR = "/home/bcml1/sigenv/_7주차_eeg/Diffu_diffusion_rev_intra2"
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# GPU 메모리 제한
def limit_gpu_memory(memory_limit_mib=14000):
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
# 전처리 Layer (필요시 사용할 수 있음)
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
# 시간(초)별 normalization 함수
def time_normalize(X):
    """
    X의 shape: (num_windows, window_size, 2, 4, 8, 8, 1)
    각 윈도우 내의 각 초(time step)에 대해 min-max normalization을 수행합니다.
    """
    X_norm = np.copy(X)
    num_windows, window_size = X.shape[0], X.shape[1]
    for i in range(num_windows):
        for t in range(window_size):
            segment = X[i, t]  # shape: (2, 4, 8, 8, 1)
            min_val = np.min(segment)
            max_val = np.max(segment)
            X_norm[i, t] = (segment - min_val) / (max_val - min_val + 1e-8)
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

# 전체 데이터 로드 후 시간대(초) 별 normalization 적용
X, Y, subjects, file_ids = load_all_data(window_size=10, step_size=1)
X = time_normalize(X)

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
# 새로 제안하는 DiffuSSM 레이어 (역확산/denoising 역할)
class DiffuSSMLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.5, bottleneck_dim=16, **kwargs):
        super(DiffuSSMLayer, self).__init__(**kwargs)
        self.hourglass_down = Dense(d_model // 2, activation=tf.nn.gelu, kernel_regularizer=l2(1e-5))
        self.hourglass_up = Dense(d_model, activation=None, kernel_regularizer=l2(1e-5))
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(d_model, return_sequences=True), merge_mode='sum'
        )
        self.dropout = Dropout(dropout_rate)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation=tf.nn.gelu, kernel_regularizer=l2(1e-5)),
            Dense(d_model, kernel_regularizer=l2(1e-5))
        ])
        self.dropout_ffn = Dropout(dropout_rate)
        self.layernorm_ffn = LayerNormalization(epsilon=1e-6)
        self.adapter = AdapterModule(d_model, bottleneck_dim)
        self.alpha = self.add_weight(shape=(1,), initializer=tf.zeros_initializer(), trainable=True, name="ssm_alpha")
    
    def call(self, inputs, training=False):
        x_hour = self.hourglass_down(inputs)
        x_hour = self.hourglass_up(x_hour)
        x_hour = self.dropout(x_hour, training=training)
        ssm_out = self.bi_gru(inputs, training=training)
        ssm_out = self.dropout(ssm_out, training=training)
        ssm_out = self.alpha * ssm_out
        x = inputs + x_hour + ssm_out
        x = self.layernorm(x)
        ffn_out = self.ffn(x)
        ffn_out = self.dropout_ffn(ffn_out, training=training)
        x = self.layernorm_ffn(x + ffn_out)
        x = self.adapter(x)
        return x

# =============================================================================
# 최종 모델: TimeDistributed로 각 1초 segment 처리 후 diffusion noise 추가 및 DiffuSSM 레이어 적용
def build_model(input_shape=(10, 2, 4, 8, 8, 1), n_layers=2, d_ff=512, p_drop=0.5, d_model=64, noise_std=0.1):
    inputs = Input(shape=input_shape)
    single_seg_model = build_single_segment_model(d_model)
    features_seq = TimeDistributed(single_seg_model, name="TimeDistributed_CNN")(inputs)
    
    def add_diffusion_noise(x):
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_std)
        return x + noise
    noisy_features_seq = Lambda(add_diffusion_noise, name="add_diffusion_noise")(features_seq)
    
    x = noisy_features_seq
    for _ in range(n_layers):
        x = DiffuSSMLayer(d_model, d_ff, dropout_rate=p_drop)(x)
    x = Flatten()(x)
    outputs = Dense(4, activation="softmax", kernel_regularizer=l2(1e-5))(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# =============================================================================
# Intra-subject 방식: 각 subject 내에서 파일 단위로 train/validation/test 분리 (overlap 방지)
X_selected = X
Y_selected = Y
F_selected = np.array(file_ids)
S_selected = subjects

unique_subjects = np.unique(S_selected)
for subj in unique_subjects:
    print(f"\n========== Intra-subject: Subject = {subj} ==========")
    subj_mask = (S_selected == subj)
    X_subj = X_selected[subj_mask]
    Y_subj = Y_selected[subj_mask]
    F_subj = F_selected[subj_mask]
    
    # 파일 단위 분리: 동일 파일의 overlapping window가 한쪽에만 배정되도록 함
    unique_files = np.unique(F_subj)
    if len(unique_files) < 2:
        print(f"Subject {subj}의 파일 수가 충분하지 않아 스킵합니다.")
        continue
    
    # 전체 파일 중 30%를 test set으로 분리
    train_files, test_files = train_test_split(unique_files, test_size=0.3, random_state=42)
    # training 파일 중 추가로 25%를 validation set으로 분리
    train_files_final, val_files = train_test_split(train_files, test_size=0.25, random_state=42)
    
    train_mask = np.isin(F_subj, train_files_final)
    val_mask = np.isin(F_subj, val_files)
    test_mask = np.isin(F_subj, test_files)
    
    X_train = X_subj[train_mask]
    Y_train = Y_subj[train_mask]
    X_val = X_subj[val_mask]
    Y_val = Y_subj[val_mask]
    X_test = X_subj[test_mask]
    Y_test = Y_subj[test_mask]
    
    print(f"Subject {subj}: Train {X_train.shape[0]} windows, Val {X_val.shape[0]} windows, Test {X_test.shape[0]} windows.")
    
    subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
    os.makedirs(subj_folder, exist_ok=True)
    
    with tf.device('/CPU:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(1000).batch(64)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(64)
    
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
    
    # 학습 과정 곡선 저장
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
    
    # Test 데이터 평가
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Subject {subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    y_pred_prob = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_prob, axis=1)
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
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
