import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
RESULT_DIR = "/home/bcml1/sigenv/_6주차_ssalmuk/dual_1"
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# GPU 메모리 제한
def limit_gpu_memory(memory_limit_mib=5000):
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

limit_gpu_memory(5000)

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
# 2D mapping 함수 (각 채널의 4 밴드 데이터를 8x8 이미지로 매핑)
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
# Dynamic Graph Convolution Layer (수정: Dense 층을 __init__에서 생성)
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
# Dual-Branch Single-Segment Model (DE와 PSD 각각 분리 처리 후 Dynamic GCN 적용)
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
# Adaptive Transformer Layer with Adapter Module
class AdaptiveTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.5, bottleneck_dim=16, **kwargs):
        super(AdaptiveTransformerLayer, self).__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation=tf.nn.gelu, kernel_regularizer=l2(1e-5)),
            Dense(d_model, kernel_regularizer=l2(1e-5))
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.adapter = AdapterModule(d_model, bottleneck_dim)
    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        out2 = self.adapter(out2)
        return out2

# =============================================================================
# 최종 모델: TimeDistributed로 각 1초 segment를 처리한 후 Adaptive Transformer 적용
def build_model(input_shape=(10, 2, 4, 8, 8, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.5, d_model=64):
    inputs = Input(shape=input_shape)
    single_seg_model = build_single_segment_model(d_model)
    features_seq = TimeDistributed(single_seg_model, name="TimeDistributed_CNN")(inputs)
    x = features_seq
    for _ in range(n_layers):
        x = AdaptiveTransformerLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)(x)
    x = Flatten()(x)
    outputs = Dense(4, activation="softmax", kernel_regularizer=l2(1e-5))(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# =============================================================================
# Fine tuning 분리 함수: 각 테스트 파일별로, 전체 윈도우 중 중간 20% 구간을 fine tuning으로,
# 나머지를 test 데이터로 분리 (윈도우 순서는 파일 내에서 유지)
def split_finetune_test_by_file(X_test, Y_test, F_test, ft_ratio=0.2):
    X_ft_list, X_test_final_list = [], []
    y_ft_list, y_test_final_list = [], []
    unique_files = np.unique(F_test)
    for f in unique_files:
        indices = np.where(F_test == f)[0]
        n = len(indices)
        start = int(n * 0.4)
        end = int(n * 0.6)
        ft_idx = indices[start:end]
        test_idx = np.concatenate([indices[:start], indices[end:]])
        X_ft_list.append(X_test[ft_idx])
        y_ft_list.append(Y_test[ft_idx])
        X_test_final_list.append(X_test[test_idx])
        y_test_final_list.append(Y_test[test_idx])
    X_ft = np.concatenate(X_ft_list, axis=0)
    y_ft = np.concatenate(y_ft_list, axis=0)
    X_test_final = np.concatenate(X_test_final_list, axis=0)
    y_test_final = np.concatenate(y_test_final_list, axis=0)
    return X_ft, X_test_final, y_ft, y_test_final

# =============================================================================
# 테스트로 사용할 subject 범위를 직접 지정 (예: subject "01"부터 "15"까지)
test_subjects = [str(i).zfill(2) for i in range(1, 16)]
test_mask = np.isin(subjects, test_subjects)
train_valid_mask = ~test_mask

X_test_all = X[test_mask]
Y_test_all = Y[test_mask]
F_test_all = np.array(file_ids)[test_mask]

X_train_valid = X[train_valid_mask]
Y_train_valid = Y[train_valid_mask]

# train/valid split (train: 75%, valid: 25% of train_valid)
if X_train_valid.shape[0] == 0:
    raise ValueError("Train/Valid set is empty. Check your subject range selection.")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_valid, Y_train_valid, test_size=0.25, random_state=42, stratify=Y_train_valid
)

# 테스트 subject의 각 파일별로 fine tuning 데이터와 최종 test 데이터 분리
X_ft, X_test_final, y_ft, y_test_final = split_finetune_test_by_file(X_test_all, Y_test_all, F_test_all, ft_ratio=0.2)

# =============================================================================
# 모델 학습 및 평가
BATCH_SIZE = 64
initial_lr = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=100000,
    decay_rate=0.9,
    staircase=True
)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, mode='max', restore_best_weights=True)

with tf.device('/CPU:0'):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

model = build_model(input_shape=(10, 2, 4, 8, 8, 1))
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
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
train_curve_path = os.path.join(RESULT_DIR, "training_curves.png")
plt.savefig(train_curve_path)
plt.close()
print(f"Training curves saved to {train_curve_path}")

# Fine tuning 단계: 테스트 파일에서 분리한 fine tuning 데이터를 사용
ft_dataset = tf.data.Dataset.from_tensor_slices((X_ft, y_ft)).shuffle(1000).batch(BATCH_SIZE)
print(f"Fine tuning with {X_ft.shape[0]} samples.")
model.fit(ft_dataset, epochs=20)

# 최종 테스트 평가: 나머지 데이터 사용
test_dataset_final = tf.data.Dataset.from_tensor_slices((X_test_final, y_test_final)).batch(BATCH_SIZE)
test_loss, test_acc = model.evaluate(test_dataset_final)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

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
cm_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to {cm_path}")

report_path = os.path.join(RESULT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"Classification report saved to {report_path}")

model_save_path = os.path.join(RESULT_DIR, "model_eeg_cnn_transformer.keras")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
