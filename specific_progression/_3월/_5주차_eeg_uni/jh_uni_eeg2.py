import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, Conv2D, Dense, LayerNormalization, 
                                     Dropout, Flatten, Input, MaxPooling2D, 
                                     Concatenate, Lambda)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures

# =============================================================================
# 결과 저장 경로 설정
RESULT_DIR = "/home/bcml1/sigenv/_5주차_eeg_uni/uni_1"
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# GPU 메모리 제한 (옵션)
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
# 전처리 Layer 정의 (Keras Layer)
class PreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # NaN, Inf 처리
        x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        x = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
        # 각 샘플별 min-max scaling (배치 차원 제외 모든 축)
        min_val = tf.reduce_min(x, axis=list(range(1, len(x.shape))), keepdims=True)
        max_val = tf.reduce_max(x, axis=list(range(1, len(x.shape))), keepdims=True)
        range_val = max_val - min_val + 1e-8
        return (x - min_val) / range_val

# =============================================================================
# DE feature 2D mapping 함수 (기존과 동일)
def map_channels_to_2d(de_segment):
    """
    de_segment: numpy array, shape (62, 4) -> 62채널, 4밴드 (예: 알파, 베타, 감마, 세타)
    각 밴드별로 62개 값을 64개(8x8)로 패딩 후 재구성하여,
    (8,8) 이미지를 생성하고 밴드를 stack.
    반환: numpy array, shape (8,8,4)
    """
    num_channels, num_bands = de_segment.shape  # (62, 4)
    mapped = np.zeros((8, 8, num_bands))
    for band in range(num_bands):
        vec = de_segment[:, band]
        vec_padded = np.pad(vec, (0, 64 - num_channels), mode='constant')
        mapped[:, :, band] = vec_padded.reshape(8, 8)
    return mapped

# =============================================================================
# 각 파일별로 DE feature와 raw 데이터를 로드하고, 분할하는 함수 (병렬 처리를 위한 helper)
def process_file(de_file_path, raw_file_path, file_name):
    de_segments = []
    raw_segments = []
    labels = []
    try:
        de_data = np.load(de_file_path)  # shape: (62, n_seconds, 4)
    except Exception as e:
        print(f"Error loading {de_file_path}: {e}")
        return de_segments, raw_segments, labels
    try:
        raw_data = np.load(raw_file_path)  # shape: (62, T, 4)
    except Exception as e:
        print(f"Error loading {raw_file_path}: {e}")
        return de_segments, raw_segments, labels
    try:
        label = int(file_name.split('_')[-1].split('.')[0])
    except Exception as e:
        print(f"Label extraction failed for {file_name}: {e}")
        return de_segments, raw_segments, labels

    n_seconds = de_data.shape[1]
    T = raw_data.shape[1]
    expected_T = n_seconds * 200  # raw 데이터는 초당 200 sample

    # T가 예상보다 1 큰 경우 (즉, n_seconds*200 + 1)라면 마지막 잡신호 제거
    if T == expected_T + 1:
        raw_data = raw_data[:, :T-1, :]
        T = raw_data.shape[1]
        print(f"Adjusted raw data length for {file_name}: new length {T}")

    if T < expected_T:
        print(f"Warning: raw data length {T} less than expected {expected_T} in file {file_name}")
        return de_segments, raw_segments, labels

    for t in range(n_seconds):
        # DE feature branch: 2D mapping 후 (4,8,8,1) 변환
        de_segment = de_data[:, t, :]  # (62,4)
        mapped_img = map_channels_to_2d(de_segment)  # (8,8,4)
        mapped_img = np.transpose(mapped_img, (2, 0, 1))  # (4,8,8)
        mapped_img = np.expand_dims(mapped_img, axis=-1)   # (4,8,8,1)
        de_segments.append(mapped_img)

        # Raw 데이터 branch: t초에 해당하는 200 sample 추출 → (62,200,4)
        start_idx = t * 200
        end_idx = start_idx + 200
        raw_segment = raw_data[:, start_idx:end_idx, :]
        raw_segments.append(raw_segment)
        labels.append(label)

    print(f"Processed {n_seconds} segments from {file_name}.")
    return de_segments, raw_segments, labels

# =============================================================================
# DE feature와 Raw 데이터 로드 및 매칭 함수 (병렬 처리 버전)
def load_all_data_multimodal(de_base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE",
                             raw_base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data"):
    """
    - DE feature 파일: shape (62, n_seconds, 4)
      → 각 초마다 DE feature를 추출하고 2D mapping 후 (4,8,8,1)로 변환.
    - Raw 데이터 파일: shape (62, T, 4), T는 초당 200 sample (즉, n_seconds * 200라고 가정)
      → 파일을 초 단위로 분할하여 (62,200,4) segment로 사용.
      단, T가 n_seconds*200 + 1 인 경우 마지막 잡신호 1개를 제거함.
    두 데이터는 파일명(예: "label_X.npy")을 통해 매칭하며, 동일한 라벨을 부여함.
    반환:
       X_de: np.array, shape=(num_samples, 4,8,8,1)
       X_raw: np.array, shape=(num_samples, 62,200,4)
       Y: np.array, shape=(num_samples,)
    """
    X_de, X_raw, Y = [], [], []
    # DE feature는 서브폴더 "1", "2", "3", raw 데이터는 "1_npy_band", "2_npy_band", "3_npy_band"
    de_subfolders = ["1", "2", "3"]
    raw_subfolders = {"1": "1_npy_band", "2": "2_npy_band", "3": "3_npy_band"}
    tasks = []

    # 로드할 모든 파일의 경로와 파일명을 미리 리스트업
    for sub in de_subfolders:
        de_folder = os.path.join(de_base_dir, sub)
        raw_folder = os.path.join(raw_base_dir, raw_subfolders[sub])
        if not os.path.isdir(de_folder):
            continue
        for file_name in os.listdir(de_folder):
            if not file_name.endswith(".npy"):
                continue
            de_file_path = os.path.join(de_folder, file_name)
            raw_file_path = os.path.join(raw_folder, file_name)
            tasks.append((de_file_path, raw_file_path, file_name))
    
    # 병렬로 각 파일을 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_file, de_fp, raw_fp, fname) for de_fp, raw_fp, fname in tasks]
        for future in concurrent.futures.as_completed(futures):
            de_segs, raw_segs, labels = future.result()
            X_de.extend(de_segs)
            X_raw.extend(raw_segs)
            Y.extend(labels)

    X_de = np.array(X_de)
    X_raw = np.array(X_raw)
    Y = np.array(Y)
    print(f"Total samples: {X_de.shape[0]}")
    print(f"DE feature sample shape: {X_de.shape[1:]}, Raw data sample shape: {X_raw.shape[1:]}")
    return X_de, X_raw, Y

# 데이터 로드 (병렬 처리된 로드)
X_de, X_raw, Y = load_all_data_multimodal()

# =============================================================================
# 2. 모델 구성: DE feature branch (3D CNN) + Raw data branch (2D CNN) + Transformer Encoder
# [DE feature branch] 기존 3D CNN 모듈
class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.spatial_conv = Conv3D(filters, kernel_size=(1, *kernel_size), strides=(1, *strides),
                                   padding="same", activation="relu")
        self.spectral_conv = Conv3D(filters, kernel_size=(kernel_size[0], 1, 1), strides=(1, *strides),
                                    padding="same", activation="relu")
    def call(self, inputs):
        spatial_features = self.spatial_conv(inputs)
        spectral_features = self.spectral_conv(inputs)
        return spatial_features + spectral_features
    def get_config(self):
        return {
            "filters": self.spatial_conv.filters,
            "kernel_size": self.spatial_conv.kernel_size,
            "strides": self.spatial_conv.strides,
        }

class SpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialSpectralAttention, self).__init__()
        self.spatial_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")
        self.spectral_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")
    def call(self, inputs):
        spatial_mask = self.spatial_squeeze(inputs)
        spatial_output = inputs * spatial_mask
        spectral_mask = self.spectral_squeeze(inputs)
        spectral_output = inputs * spectral_mask
        return spatial_output + spectral_output
    def get_config(self):
        return {}

# [Raw 데이터 branch] 2D CNN을 이용한 특징 추출
def build_raw_cnn_branch(input_shape=(62,200,4), d_model=64):
    raw_input = Input(shape=input_shape)
    x = PreprocessLayer()(raw_input)
    x = Conv2D(16, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(32, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(d_model, activation="relu")(x)
    model = Model(inputs=raw_input, outputs=x, name="raw_cnn_branch")
    return model

# [Transformer Encoder Layer] 기존 코드와 동일
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation=tf.nn.gelu),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        return {
            "d_model": self.mha.key_dim * self.mha.num_heads,
            "n_heads": self.mha.num_heads,
            "d_ff": self.ffn.layers[0].units,
            "dropout_rate": self.dropout1.rate,
        }

# [Combined Model] 두 branch의 결과를 Concatenate 후 Transformer에 입력
def build_multimodal_model(de_input_shape=(4,8,8,1), raw_input_shape=(62,200,4),
                           n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64):
    # DE feature branch (3D CNN)
    de_input = Input(shape=de_input_shape, name="de_input")
    x_de = PreprocessLayer()(de_input)
    x_de = SpatialSpectralAttention()(x_de)
    x_de = SpatialSpectralConvModule(8, kernel_size=(3,3), strides=(3,3))(x_de)
    x_de = SpatialSpectralConvModule(16, kernel_size=(1,1), strides=(1,1))(x_de)
    x_de = SpatialSpectralConvModule(32, kernel_size=(1,1), strides=(1,1))(x_de)
    x_de = Flatten()(x_de)
    x_de = Dense(d_model, activation="relu")(x_de)
    
    # Raw 데이터 branch (2D CNN)
    raw_input = Input(shape=raw_input_shape, name="raw_input")
    raw_cnn = build_raw_cnn_branch(input_shape=raw_input_shape, d_model=d_model)
    x_raw = raw_cnn(raw_input)
    
    # 두 branch의 특징 벡터 Concatenate
    combined = Concatenate()([x_de, x_raw])
    combined_proj = Dense(d_model, activation="relu")(combined)
    
    # Transformer 입력을 위해 차원 확장 (sequence length = 1)
    x = Lambda(lambda z: tf.expand_dims(z, axis=1))(combined_proj)
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)(x)
    x = Lambda(lambda z: tf.squeeze(z, axis=1))(x)
    outputs = Dense(4, activation="softmax")(x)
    
    model = Model(inputs=[de_input, raw_input], outputs=outputs, name="multimodal_model")
    return model

model = build_multimodal_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# =============================================================================
# 데이터 분할 및 학습
X_de_train_val, X_de_test, X_raw_train_val, X_raw_test, y_train_val, y_test = train_test_split(
    X_de, X_raw, Y, test_size=0.2, random_state=42, stratify=Y
)
X_de_train, X_de_val, X_raw_train, X_raw_val, y_train, y_val = train_test_split(
    X_de_train_val, X_raw_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

BATCH_SIZE = 16
train_dataset = tf.data.Dataset.from_tensor_slices(((X_de_train, X_raw_train), y_train)).shuffle(1000).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices(((X_de_val, X_raw_val), y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(((X_de_test, X_raw_test), y_test)).batch(BATCH_SIZE)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
    # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = model.fit(train_dataset, epochs=150, validation_data=val_dataset, callbacks=callbacks)

# =============================================================================
# 학습곡선 그리기 및 저장
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
train_curve_path = os.path.join(RESULT_DIR, "training_curves_multimodal.png")
plt.savefig(train_curve_path)
plt.close()
print(f"Training curves saved to {train_curve_path}")

# =============================================================================
# 테스트 데이터 평가 및 예측
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# 예측 수행
y_pred_prob = model.predict(test_dataset)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
cm_path = os.path.join(RESULT_DIR, "confusion_matrix_multimodal.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to {cm_path}")

report_path = os.path.join(RESULT_DIR, "classification_report_multimodal.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"Classification report saved to {report_path}")

model_save_path = os.path.join(RESULT_DIR, "model_multimodal_eeg.keras")
model.save_weights(model_save_path)
print(f"Model weights saved to {model_save_path}")
