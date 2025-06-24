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
# 결과 저장 경로 설정 (모든 subject의 결과가 이 폴더 아래에 저장됩니다)
RESULT_DIR = "/home/bcml1/sigenv/_5주차_eeg_CNN+TRAN/result1_intra"
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
# 전처리 Layer (Keras Layer로 래핑)
class PreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # NaN, Inf 처리
        x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        x = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
        # per-sample min-max scaling (입력 shape: (batch, 4, 8, 8, 1))
        min_val = tf.reduce_min(x, axis=[1,2,3,4], keepdims=True)
        max_val = tf.reduce_max(x, axis=[1,2,3,4], keepdims=True)
        range_val = max_val - min_val + 1e-8
        return (x - min_val) / range_val

# =============================================================================
# 1. 데이터 전처리 함수: 각 npy 파일 내 DE feature를 초 단위로 분리 후 8x8 2D mapping
def map_channels_to_2d(de_segment):
    """
    de_segment: numpy array, shape (62, 4) → 62채널, 4밴드 (알파, 베타, 감마, 세타)
    각 밴드별로 62개 값을 64개(8x8)로 패딩 후 재구성하여 (8,8) 이미지를 생성하고,
    밴드를 스택하여 반환합니다. (반환 shape: (8,8,4))
    """
    num_channels, num_bands = de_segment.shape
    mapped = np.zeros((8, 8, num_bands))
    for band in range(num_bands):
        vec = de_segment[:, band]
        vec_padded = np.pad(vec, (0, 64 - num_channels), mode='constant')
        mapped[:, :, band] = vec_padded.reshape(8, 8)
    return mapped

def load_subject_data(subject, base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE"):
    """
    subject: 문자열로 주제 번호 (예: "1" 또는 "15"). 파일명은 subject 번호로 시작한다고 가정합니다.
    base_dir: DE feature 파일들이 저장된 기본 폴더 (하위 폴더 "1", "2", "3" 존재)
    각 파일은 shape (62, n_seconds, 4)를 가지며, 초마다 데이터를 추출하여
    최종 입력 shape (4, 8, 8, 1)로 변환합니다.
    라벨은 파일명 마지막의 "label_X.npy"에서 X를 정수형으로 추출합니다.
    """
    X, Y = [], []
    for subfolder in ["1", "2", "3"]:
        folder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            # 파일명이 subject 번호로 시작하는지 확인 (예: "1_" 또는 "15_")
            if not file_name.startswith(f"{subject}_"):
                continue
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
            except Exception as e:
                print(f"Label extraction failed for {file_name}: {e}")
                continue
            n_seconds = de_data.shape[1]
            for t in range(n_seconds):
                segment = de_data[:, t, :]  # (62, 4)
                mapped_img = map_channels_to_2d(segment)  # (8,8,4)
                # 원하는 입력 shape: (밴드, height, width, channel=1) → (4, 8, 8, 1)
                mapped_img = np.transpose(mapped_img, (2, 0, 1))
                mapped_img = np.expand_dims(mapped_img, axis=-1)
                X.append(mapped_img)
                Y.append(label)
            print(f"Subject {subject}: Loaded {n_seconds} segments from {file_path} with label {label}.")
    X = np.array(X)
    Y = np.array(Y)
    print(f"Subject {subject}: Total samples: {X.shape[0]}, each sample shape: {X.shape[1:]}")
    return X, Y

# =============================================================================
# 2. 모델 구성: 3D CNN + Transformer Encoder

# Spatial-Spectral Convolution Module
class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.spatial_conv = Conv3D(filters, kernel_size=(1, *kernel_size), strides=(1, *strides),
                                   padding="same", activation="relu")
        # spectral_conv: stride를 (1, *strides)로 하여 두 branch의 출력 shape을 맞춤
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
        spectral_mask = self.spectral_squeeze(inputs)
        return inputs * spatial_mask + inputs * spectral_mask
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

# 최종 모델: 3D CNN + Transformer Encoder 결합
def build_model(input_shape=(4,8,8,1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64):
    """
    입력 shape: (4, 8, 8, 1) → 4 밴드, 8×8 spatial, 채널 1
    """
    inputs = Input(shape=input_shape)
    x = PreprocessLayer()(inputs)
    x = SpatialSpectralAttention()(x)
    x = SpatialSpectralConvModule(8, kernel_size=(3,3), strides=(3,3))(x)
    x = SpatialSpectralConvModule(16, kernel_size=(1,1), strides=(1,1))(x)
    x = SpatialSpectralConvModule(32, kernel_size=(1,1), strides=(1,1))(x)
    x = Flatten()(x)
    x = Dense(d_model, activation="relu")(x)
    # 차원 확장 및 축소는 Lambda 레이어로 처리
    x = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z, axis=1))(x)  # (batch, 1, d_model)
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)(x)
    x = tf.keras.layers.Lambda(lambda z: tf.squeeze(z, axis=1))(x)  # (batch, d_model)
    outputs = Dense(4, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# =============================================================================
# 각 subject별 학습 (intra-subject training)
# subject ID는 파일명이 "subject_..." 형식으로 시작한다고 가정 (예: "1_..." for subject 1)
subjects = [str(i) for i in range(1, 33)]  # 예: 1부터 32번 subject

BATCH_SIZE = 16

for subj in subjects:
    print(f"\n========== Training subject: {subj} ==========")
    # subject별 데이터 로드
    X_subj, Y_subj = load_subject_data(subj)
    if X_subj.shape[0] == 0:
        print(f"Subject {subj}: No data found. Skipping.")
        continue

    # train/validation/test 분할 (전체 20% test, 나머지 80% 중 25%를 validation → 60%/20%/20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_subj, Y_subj, test_size=0.2, random_state=42, stratify=Y_subj
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # subject 전용 결과 폴더 생성 (예: s01, s15)
    subj_folder = os.path.join(RESULT_DIR, f"s{subj.zfill(2)}")
    os.makedirs(subj_folder, exist_ok=True)

    # 데이터셋 생성
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    # 모델 생성 및 컴파일
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        #ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    # 학습
    history = model.fit(train_dataset, epochs=150, validation_data=val_dataset, callbacks=callbacks)

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
    train_curve_path = os.path.join(subj_folder, "training_curves.png")
    plt.savefig(train_curve_path)
    plt.close()
    print(f"Subject {subj}: Training curves saved to {train_curve_path}")

    # 테스트 평가 및 예측
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Subject {subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    y_pred_prob = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_prob, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    # confusion matrix 저장
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(subj_folder, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Subject {subj}: Confusion matrix saved to {cm_path}")
    # classification report 저장
    report_path = os.path.join(subj_folder, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Subject {subj}: Classification report saved to {report_path}")

    # 모델 저장 (전체 모델 저장: Keras SavedModel 형식)
    model_save_path = os.path.join(subj_folder, "model_eeg_cnn_transformer.keras")
    model.save(model_save_path)
    print(f"Subject {subj}: Model saved to {model_save_path}")
