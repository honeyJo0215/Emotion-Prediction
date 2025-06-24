import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, LayerNormalization, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 결과 저장 경로 설정
RESULT_DIR = "/home/bcml1/sigenv/_5주차_eeg_CNN+TRAN/result1_inter"
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
        # per-sample min-max scaling (입력 shape: (batch, 4,8,8,1))
        min_val = tf.reduce_min(x, axis=[1,2,3,4], keepdims=True)
        max_val = tf.reduce_max(x, axis=[1,2,3,4], keepdims=True)
        range_val = max_val - min_val + 1e-8
        return (x - min_val) / range_val

# =============================================================================
# 1. 데이터 전처리: 각 npy 파일 내 DE feature를 초 단위로 분리하고,
# 62채널 데이터를 8x8 2D mapping (패딩 포함)으로 변환
# 입력: de_feature 배열 shape = (62, n_seconds, 4)
# 출력: 각 초마다 2D mapping된 데이터 shape = (4, 8,8, 1)
def map_channels_to_2d(de_segment):
    """
    de_segment: numpy array, shape (62, 4) --> 62채널, 4밴드 (알파, 베타, 감마, 세타)
    각 밴드별로 62개 값을 64개(8x8)로 패딩 후 재구성하여,
    최종적으로 (8,8) 이미지를 생성한 후 밴드들을 스택.
    반환: numpy array, shape (8,8,4)
    """
    num_channels, num_bands = de_segment.shape  # (62, 4)
    mapped = np.zeros((8, 8, num_bands))
    for band in range(num_bands):
        vec = de_segment[:, band]
        vec_padded = np.pad(vec, (0, 64 - num_channels), mode='constant')
        mapped[:, :, band] = vec_padded.reshape(8, 8)
    return mapped  # (8,8,4)

def load_all_data(base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE"):
    """
    base_dir 내의 서브폴더 (예: "1", "2", "3")에 있는 모든 npy 파일을 순회.
    각 npy 파일은 shape (62, n_seconds, 4)이며,
    각 초마다 62채널 데이터를 8x8 2D mapping 후, (밴드, 8,8, 1) 형태로 변환.
    파일명에서 "label_X.npy" 형태로 라벨 추출 (정수형).
    반환: X (np.array, shape = (num_samples, 4,8,8,1)), Y (np.array, shape = (num_samples,))
    """
    X, Y = [], []
    for subfolder in ["1", "2", "3"]:
        folder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            if not file_name.endswith(".npy"):
                continue
            file_path = os.path.join(folder_path, file_name)
            try:
                de_data = np.load(file_path)  # shape: (62, n_seconds, 4)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            try:
                label = int(file_name.split('_')[-1].split('.')[0])
            except:
                print(f"Label extraction failed for {file_name}")
                continue
            n_seconds = de_data.shape[1]
            for t in range(n_seconds):
                segment = de_data[:, t, :]  # (62, 4)
                mapped_img = map_channels_to_2d(segment)  # (8,8,4)
                # 원하는 입력 shape: (밴드, height, width, channel=1) => (4,8,8,1)
                mapped_img = np.transpose(mapped_img, (2, 0, 1))
                mapped_img = np.expand_dims(mapped_img, axis=-1)
                X.append(mapped_img)
                Y.append(label)
            print(f"Loaded {n_seconds} segments from {file_path} with label {label}.")
    X = np.array(X)
    Y = np.array(Y)
    print(f"Total samples: {X.shape[0]}, each sample shape: {X.shape[1:]}")
    return X, Y

# 데이터 로드
X, Y = load_all_data()

# =============================================================================
# 2. 모델 구성: 3D CNN + Transformer Encoder

# Spatial-Spectral Convolution Module
class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.spatial_conv = Conv3D(filters, kernel_size=(1, *kernel_size), strides=(1, *strides),
                                   padding="same", activation="relu")
        # spectral_conv의 stride를 (1, *strides)로 설정하여 두 분기의 출력 shape을 맞춤
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

# Spatial and Spectral Attention Branch
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

# Transformer Encoder Layer
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
        return self.layernorm2(inputs + ffn_output)
    def get_config(self):
        return {
            "d_model": self.mha.key_dim * self.mha.num_heads,
            "n_heads": self.mha.num_heads,
            "d_ff": self.ffn.layers[0].units,
            "dropout_rate": self.dropout1.rate,
        }

# 최종 모델: 3D CNN 블록과 Transformer Encoder를 결합
def build_model(input_shape=(4,8,8,1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64):
    """
    입력 shape: (4,8,8,1) -> 4 밴드, 8x8 spatial, 채널 차원 1
    """
    inputs = Input(shape=input_shape)
    # PreprocessLayer를 사용하여 전처리 수행
    x = PreprocessLayer()(inputs)
    x = SpatialSpectralAttention()(x)
    x = SpatialSpectralConvModule(8, kernel_size=(3,3), strides=(3,3))(x)
    x = SpatialSpectralConvModule(16, kernel_size=(1,1), strides=(1,1))(x)
    x = SpatialSpectralConvModule(32, kernel_size=(1,1), strides=(1,1))(x)
    x = Flatten()(x)  # (batch, feature_dim)
    x = Dense(d_model, activation="relu")(x)  # (batch, d_model)
    # Transformer encoder를 sequence로 처리하기 위해 Lambda 레이어로 차원 확장 (길이=1)
    x = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z, axis=1))(x)  # (batch, 1, d_model)
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)(x)
    # 차원 축소도 Lambda 레이어로 처리
    x = tf.keras.layers.Lambda(lambda z: tf.squeeze(z, axis=1))(x)  # (batch, d_model)
    outputs = Dense(4, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 모델 생성 및 컴파일
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# =============================================================================
# 3. 데이터 분할 및 학습
# 전체 데이터를 train/validation/test 세트로 분리
# (예: 전체의 20%를 test로 분리한 후, 나머지 80% 중 25%를 validation으로 사용 → train:60%, val:20%, test:20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

BATCH_SIZE = 16
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
    #ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
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
train_curve_path = os.path.join(RESULT_DIR, "training_curves.png")
plt.savefig(train_curve_path)
plt.close()
print(f"Training curves saved to {train_curve_path}")

# =============================================================================
# 테스트 데이터 평가 및 예측
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# 테스트 데이터에 대한 예측
y_pred_prob = model.predict(test_dataset)
y_pred = np.argmax(y_pred_prob, axis=1)

# confusion matrix 및 classification report 출력
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)

# confusion matrix 시각화 및 저장
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
cm_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to {cm_path}")

# classification report 저장
report_path = os.path.join(RESULT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"Classification report saved to {report_path}")

# =============================================================================
# 모델 저장 (모델 가중치)
model_save_path = os.path.join(RESULT_DIR, "model_eeg_cnn_transformer.keras")
model.save_weights(model_save_path)
print(f"Model weights saved to {model_save_path}")
