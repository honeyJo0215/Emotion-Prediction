import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, Conv2D, Dense, LayerNormalization, 
                                     Dropout, Flatten, Input, MaxPooling2D, 
                                     Concatenate, Reshape, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 결과 저장 경로 기본 디렉토리
BASE_RESULT_DIR = "/home/bcml1/sigenv/_5주차_eeg_uni/uni_1"
os.makedirs(BASE_RESULT_DIR, exist_ok=True)

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
    de_segment: numpy array, shape (62, 4) -> 62채널, 4밴드
    각 밴드별로 62개 값을 64개(8x8)로 패딩 후 재구성하여,
    (8,8) 이미지를 생성하고 밴드를 stack.
    반환: numpy array, shape (8,8,4)
    """
    num_channels, num_bands = de_segment.shape
    mapped = np.zeros((8, 8, num_bands), dtype=np.float32)
    for band in range(num_bands):
        vec = de_segment[:, band]
        vec_padded = np.pad(vec, (0, 64 - num_channels), mode='constant')
        mapped[:, :, band] = vec_padded.reshape(8, 8)
    return mapped

# =============================================================================
# 파일 목록 생성 함수 (전체 데이터를 반환)
def get_task_list(de_base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE",
                  raw_base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data"):
    tasks = []
    de_subfolders = ["1", "2", "3"]
    raw_subfolders = {"1": "1_npy_band", "2": "2_npy_band", "3": "3_npy_band"}
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
    return tasks

# =============================================================================
# 데이터 제너레이터 (코드는 기존과 동일)
def data_generator(task_list):
    for de_file_path, raw_file_path, file_name in task_list:
        try:
            de_data = np.load(de_file_path, mmap_mode='r')  # shape: (62, n_seconds, 4)
            raw_data = np.load(raw_file_path, mmap_mode='r')  # shape: (62, T, 4)
        except Exception as e:
            print(f"Error loading files {file_name}: {e}")
            continue
        try:
            label = int(file_name.split('_')[-1].split('.')[0])
        except Exception as e:
            print(f"Label extraction failed for {file_name}: {e}")
            continue

        n_seconds = de_data.shape[1]
        T = raw_data.shape[1]
        expected_T = n_seconds * 200  # raw 데이터: 초당 200 sample

        if T == expected_T + 1:
            raw_data = raw_data[:, :T-1, :]
            T = raw_data.shape[1]
            print(f"Adjusted raw data length for {file_name}: new length {T}")

        if T < expected_T:
            print(f"Warning: raw data length {T} less than expected {expected_T} in {file_name}")
            continue

        for t in range(n_seconds):
            de_segment = de_data[:, t, :]  # (62, 4)
            mapped_img = map_channels_to_2d(de_segment)  # (8,8,4)
            mapped_img = np.transpose(mapped_img, (2, 0, 1))  # (4,8,8)
            mapped_img = np.expand_dims(mapped_img, axis=-1)   # (4,8,8,1)

            start_idx = t * 200
            end_idx = start_idx + 200
            raw_segment = raw_data[:, start_idx:end_idx, :]  # (62,200,4)

            yield ((mapped_img.astype(np.float32), raw_segment.astype(np.float32)),
                   np.int32(label))

# =============================================================================
# 기존 SpatialSpectral 모듈 및 Transformer Encoder Layer (변경 없음)
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

# =============================================================================
# Positional Embedding Layer
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        # input_shape: (batch, sequence_length, d_model)
        self.sequence_length = input_shape[1]
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(self.sequence_length, self.d_model),
            initializer=tf.random_normal_initializer(),
            trainable=True
        )
        super(PositionalEmbedding, self).build(input_shape)

    def call(self, x):
        return x + self.pos_embedding

# =============================================================================
# 개선된 멀티모달 모델 구성 함수
def build_multimodal_model(de_input_shape=(4,8,8,1), raw_input_shape=(62,200,4),
                           n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64, num_classes=4):
    # DE branch
    de_input = Input(shape=de_input_shape, name="de_input")
    x_de = SpatialSpectralAttention()(de_input)
    x_de = SpatialSpectralConvModule(8, kernel_size=(3,3), strides=(3,3))(x_de)
    x_de = SpatialSpectralConvModule(16, kernel_size=(1,1), strides=(1,1))(x_de)
    x_de = SpatialSpectralConvModule(32, kernel_size=(1,1), strides=(1,1))(x_de)
    # x_de shape: (batch, 4, 3, 3, 32)
    x_de = Reshape((4*3*3, 32))(x_de)  # (batch, 36, 32)
    x_de = Dense(d_model, activation="relu")(x_de)  # (batch, 36, d_model)
    
    # Raw branch
    raw_input = Input(shape=raw_input_shape, name="raw_input")
    x_raw = PreprocessLayer()(raw_input)
    x_raw = Conv2D(16, kernel_size=(3,3), padding="same", activation="relu")(x_raw)   # (62,200,16)
    x_raw = MaxPooling2D(pool_size=(2,2))(x_raw)  # 약 (31,100,16)
    x_raw = Conv2D(32, kernel_size=(3,3), padding="same", activation="relu")(x_raw)   # (31,100,32)
    x_raw = MaxPooling2D(pool_size=(2,2))(x_raw)  # 약 (16,50,32)
    x_raw = Conv2D(64, kernel_size=(3,3), padding="same", activation="relu")(x_raw)   # (16,50,64)
    x_raw = MaxPooling2D(pool_size=(2,2))(x_raw)  # 약 (8,25,64)
    x_raw = Reshape((-1, x_raw.shape[-1]))(x_raw)  # (batch, 8*25=200, 64)
    x_raw = Dense(d_model, activation="relu")(x_raw)  # (batch, 200, d_model)
    
    # 두 branch의 토큰 시퀀스를 결합 (동적 토큰 수)
    combined_tokens = Concatenate(axis=1)([x_de, x_raw])
    combined_tokens = Dropout(p_drop)(combined_tokens)

    # Positional Embedding 추가
    x = PositionalEmbedding(d_model=d_model)(combined_tokens)

    # Transformer Encoder Layers
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)(x)
    
    # 토큰 시퀀스에 대해 Global Average Pooling 적용
    x = GlobalAveragePooling1D()(x)
    
    # Classification head
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=[de_input, raw_input], outputs=outputs, name="multimodal_model")
    return model

# =============================================================================
# 전체 task 목록 로드
all_tasks = get_task_list()
print(f"Total files found: {len(all_tasks)}")

# subject 리스트: s01 ~ s15
subjects = [f"s{str(i).zfill(2)}" for i in range(1, 16)]

for subject in subjects:
    print("\n==========================================")
    print(f"Processing subject: {subject}")
    
    # subject에 해당하는 task만 필터링
    subject_tasks = [task for task in all_tasks if task[2].startswith(f"{subject}_")]
    print(f"Total files for {subject}: {len(subject_tasks)}")
    
    # task가 부족하면 건너뛰기
    if len(subject_tasks) < 5:
        print(f"Not enough files for {subject}. Skipping...")
        continue
    
    # intra-subject 데이터 분할 (train: 60%, val: 20%, test: 20%)
    train_tasks, test_tasks = train_test_split(subject_tasks, test_size=0.2, random_state=42)
    train_tasks, val_tasks = train_test_split(train_tasks, test_size=0.25, random_state=42)
    print(f"Train files: {len(train_tasks)}, Val files: {len(val_tasks)}, Test files: {len(test_tasks)}")
    
    # tf.data.Dataset 생성
    output_types = ((tf.float32, tf.float32), tf.int32)
    output_shapes = (((tf.TensorShape([4,8,8,1]), tf.TensorShape([62,200,4])), tf.TensorShape([])))
    
    train_dataset = tf.data.Dataset.from_generator(lambda: data_generator(train_tasks),
                                                   output_types=output_types,
                                                   output_shapes=output_shapes)
    val_dataset = tf.data.Dataset.from_generator(lambda: data_generator(val_tasks),
                                                 output_types=output_types,
                                                 output_shapes=output_shapes)
    test_dataset = tf.data.Dataset.from_generator(lambda: data_generator(test_tasks),
                                                  output_types=output_types,
                                                  output_shapes=output_shapes)
    
    BATCH_SIZE = 16
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # =============================================================================
    # 모델 구성 및 컴파일
    model = build_multimodal_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    
    # =============================================================================
    # 모델 학습
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
    ]
    
    history = model.fit(train_dataset, epochs=150, validation_data=val_dataset, callbacks=callbacks)
    
    # =============================================================================
    # 결과 저장을 위한 subject 전용 디렉토리 생성
    subject_result_dir = os.path.join(BASE_RESULT_DIR, subject)
    os.makedirs(subject_result_dir, exist_ok=True)
    
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
    train_curve_path = os.path.join(subject_result_dir, "training_curves_multimodal.png")
    plt.savefig(train_curve_path)
    plt.close()
    print(f"Training curves saved to {train_curve_path}")
    
    # =============================================================================
    # 테스트 데이터 평가 및 예측
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    y_pred_prob = model.predict(test_dataset)
    y_pred = tf.argmax(y_pred_prob, axis=1).numpy()
    
    y_true = []
    for batch_labels in test_dataset.map(lambda x, y: y):
        y_true.extend(batch_labels.numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(subject_result_dir, "confusion_matrix_multimodal.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    report_path = os.path.join(subject_result_dir, "classification_report_multimodal.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
    model_save_path = os.path.join(subject_result_dir, "model_multimodal_eeg.keras")
    model.save_weights(model_save_path)
    print(f"Model weights saved to {model_save_path}")

print("\nAll subjects processed.")
