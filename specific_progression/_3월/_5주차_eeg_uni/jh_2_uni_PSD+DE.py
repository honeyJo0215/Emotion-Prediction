import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, LayerNormalization, Dropout, Flatten, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 결과 저장 경로 설정 (inter subject)
RESULT_DIR = "/home/bcml1/sigenv/_5주차_eeg_uni/uni_2_1_PSD+DE"
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
        # 입력의 정적 rank를 사용하여 per-sample min-max scaling 수행
        axis = list(range(1, inputs.shape.rank))
        min_val = tf.reduce_min(x, axis=axis, keepdims=True)
        max_val = tf.reduce_max(x, axis=axis, keepdims=True)
        range_val = max_val - min_val + 1e-8
        return (x - min_val) / range_val

# =============================================================================
# 2D mapping 함수 (공통)
def map_channels_to_2d(segment):
    """
    segment: numpy array, shape (62, 4) → 62채널, 4밴드 (알파, 베타, 감마, 세타)
    각 밴드별로 62개 값을 64개(8x8)로 패딩 후 재구성하여 (8,8) 이미지를 생성하고,
    밴드를 스택하여 반환 (반환 shape: (8,8,4))
    """
    num_channels, num_bands = segment.shape
    mapped = np.zeros((8, 8, num_bands))
    for band in range(num_bands):
        vec = segment[:, band]
        vec_padded = np.pad(vec, (0, 64 - num_channels), mode='constant')
        mapped[:, :, band] = vec_padded.reshape(8, 8)
    return mapped

# subject normalize
def subject_normalize(X, S):
    """
    X: numpy array, shape (num_samples, 2, 4, 8, 8, 1)
       - 첫 번째 축: 모달리티 (0: DE, 1: PSD)
    S: numpy array, shape (num_samples,), subject id 정보
    각 subject에 대해 각 모달리티의 전체 평균을 구해, 그 평균을 빼서 값의 평균을 0으로 맞춥니다.
    """
    X_norm = np.copy(X)
    unique_subjects = np.unique(S)
    for subj in unique_subjects:
        subj_idx = np.where(S == subj)[0]
        # DE modality (index 0)
        mean_de = np.mean(X[subj_idx, 0, ...])
        X_norm[subj_idx, 0, ...] = X_norm[subj_idx, 0, ...] - mean_de
        # PSD modality (index 1)
        mean_psd = np.mean(X[subj_idx, 1, ...])
        X_norm[subj_idx, 1, ...] = X_norm[subj_idx, 1, ...] - mean_psd
    return X_norm

# =============================================================================
# 데이터 로드 함수: DE와 PSD 데이터를 함께 로드하여, 
# 각 1초(=한 segment)의 DE와 PSD 이미지를 각각 2D mapping 후, 
# 두 modality를 sequence로 stacking → 최종 입력 shape: (2,4,8,8,1)
def load_all_data(de_base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE",
                  psd_base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_PSD"):
    X, Y, S = [], [], []
    # subfolder: "1", "2", "3"
    for subfolder in ["1", "2", "3"]:
        de_folder = os.path.join(de_base_dir, subfolder)
        psd_folder = os.path.join(psd_base_dir, subfolder)
        if not os.path.isdir(de_folder) or not os.path.isdir(psd_folder):
            continue
        for file_name in os.listdir(de_folder):
            if not file_name.endswith(".npy"):
                continue
            de_file_path = os.path.join(de_folder, file_name)
            psd_file_path = os.path.join(psd_folder, file_name)  # 동일 파일명으로 존재한다고 가정
            try:
                de_data = np.load(de_file_path)   # shape: (62, n_seconds, 4)
                psd_data = np.load(psd_file_path)  # shape: (62, n_seconds, 4)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue
            try:
                # label 추출: 파일명 마지막 토큰에서 정수형 label (예: "label_1.npy" → 1)
                label = int(file_name.split('_')[-1].split('.')[0])
            except:
                print(f"Label extraction failed for {file_name}")
                continue
            # subject id 추출: 파일명 첫번째 토큰 (예: "1_20160518_sample_01_label_1.npy" → "1")
            subject_id = file_name.split('_')[0]
            n_seconds = de_data.shape[1]
            for t in range(n_seconds):
                de_segment = de_data[:, t, :]   # (62, 4)
                psd_segment = psd_data[:, t, :]   # (62, 4)
                # 2D mapping: (8,8,4)
                de_mapped = map_channels_to_2d(de_segment)
                psd_mapped = map_channels_to_2d(psd_segment)
                # 변환: (4,8,8) 후 채널 축 추가 → (4,8,8,1)
                de_img = np.transpose(de_mapped, (2, 0, 1))
                de_img = np.expand_dims(de_img, axis=-1)
                psd_img = np.transpose(psd_mapped, (2, 0, 1))
                psd_img = np.expand_dims(psd_img, axis=-1)
                # 두 modality(각 1초)를 시퀀스로 stacking: 첫 번째는 DE, 두 번째는 PSD
                sample = np.stack([de_img, psd_img], axis=0)  # shape: (2, 4, 8, 8, 1)
                X.append(sample)
                Y.append(label)
                S.append(subject_id)
            print(f"Loaded {n_seconds} segments from {file_name} with label {label} and subject {subject_id}.")
    X = np.array(X)
    Y = np.array(Y)
    S = np.array(S)
    print(f"Total samples: {X.shape[0]}, each sample shape: {X.shape[1:]}")
    return X, Y, S

# 전체 데이터 로드 (DE와 PSD 모두)
X, Y, subjects = load_all_data()
# subject별 평균 값을 0으로 맞추는 전처리 적용
X = subject_normalize(X, subjects)

# =============================================================================
# 2. 모델 구성: TimeDistributed CNN (공유 CNN branch) + Transformer Encoder
# 입력 shape: (2, 4,8,8,1) → 시퀀스 길이 2 (각 1초, modality: DE와 PSD)

# Spatial-Spectral Convolution Module (변경 없음)
class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.spatial_conv = Conv3D(filters, kernel_size=(1, *kernel_size), strides=(1, *strides),
                                   padding="same", activation="relu",
                                   kernel_regularizer=l2(1e-4))
        self.spectral_conv = Conv3D(filters, kernel_size=(kernel_size[0], 1, 1), strides=(1, *strides),
                                    padding="same", activation="relu",
                                    kernel_regularizer=l2(1e-4))
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

# Spatial and Spectral Attention Branch (변경 없음)
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

# TransformerEncoderLayer (내부 FFN에도 L2 정규화 적용)
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation=tf.nn.gelu, kernel_regularizer=l2(1e-4)),
            Dense(d_model, kernel_regularizer=l2(1e-4))
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
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            "d_model": self.mha.key_dim * self.mha.num_heads,
            "n_heads": self.mha.num_heads,
            "d_ff": self.ffn.layers[0].units,
            "dropout_rate": self.dropout1.rate,
        }

# CNN branch를 커스텀 레이어로 정의 (각 타임스텝마다 적용)
class CNNBranchLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(CNNBranchLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.preprocess = PreprocessLayer()
        self.attention = SpatialSpectralAttention()
        self.conv1 = SpatialSpectralConvModule(8, kernel_size=(3,3), strides=(3,3))
        self.conv2 = SpatialSpectralConvModule(16, kernel_size=(1,1), strides=(1,1))
        self.conv3 = SpatialSpectralConvModule(32, kernel_size=(1,1), strides=(1,1))
        self.flatten = Flatten()
        # L2 정규화 적용
        self.dense = Dense(d_model, activation="relu", kernel_regularizer=l2(1e-4))
    
    def call(self, x):
        # x shape: (4,8,8,1)
        x = self.preprocess(x)
        x = self.attention(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 5:
            return (input_shape[0], self.d_model)
        else:
            return (self.d_model,)

# 최종 모델: DE와 PSD를 각각 다른 CNN branch에서 처리한 후 Transformer Encoder에 입력
def build_model(input_shape=(2, 4, 8, 8, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64):
    """
    입력 shape: (2, 4, 8, 8, 1)
      - 2: 모달리티 (첫 번째: DE, 두 번째: PSD)
      - 4: 채널 (2D mapping된 후 채널 수)
      - 8x8: 공간 정보
      - 1: 단일 채널
    각 모달리티는 별도의 CNN branch를 통해 (d_model,) 벡터로 변환되고,
    이 두 벡터를 시퀀스로 stacking하여 Transformer Encoder에 투입합니다.
    """
    inputs = Input(shape=input_shape)  # shape: (batch, 2, 4, 8, 8, 1)
    
    # 모달리티 분리
    de_input = Lambda(lambda x: x[:, 0, ...], name="DE_input")(inputs)   # (batch, 4,8,8,1)
    psd_input = Lambda(lambda x: x[:, 1, ...], name="PSD_input")(inputs)  # (batch, 4,8,8,1)
    
    # 각 모달리티 별 CNN branch
    cnn_de = CNNBranchLayer(d_model, name="CNN_DE")
    cnn_psd = CNNBranchLayer(d_model, name="CNN_PSD")
    
    feature_de = cnn_de(de_input)    # (batch, d_model)
    feature_psd = cnn_psd(psd_input)   # (batch, d_model)
    
    # 두 feature를 시퀀스로 stacking: (batch, 2, d_model)
    combined_features = Lambda(lambda x: tf.stack(x, axis=1), name="Stack_Features")([feature_de, feature_psd])
    
    # Transformer Encoder layers 적용
    x = combined_features
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)(x)
    # Flatten하여 최종 feature vector 생성 (batch, 2*d_model)
    x = Flatten()(x)
    outputs = Dense(4, activation="softmax", kernel_regularizer=l2(1e-4))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# =============================================================================
# Inter-Subject 학습: leave-one-subject-out 방식
X_all, Y_all, S_all = X, Y, subjects
unique_subjects = np.unique(S_all)

BATCH_SIZE = 16

# 하이퍼파라미터: 초기 학습률
initial_lr = 1e-4

# 내부 learning rate schedule 적용 (ExponentialDecay)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,    # 필요에 따라 조정 가능
    decay_rate=0.9,
    staircase=True
)

for subj in unique_subjects:
    print(f"\n========== Inter-Subject: Test subject = {subj} ==========")
    # test set: 해당 subject의 데이터, train/valid: 나머지 subject의 데이터
    test_mask = (S_all == subj)
    train_valid_mask = (S_all != subj)
    
    X_test = X_all[test_mask]
    Y_test = Y_all[test_mask]
    X_train_valid = X_all[train_valid_mask]
    Y_train_valid = Y_all[train_valid_mask]
    
    # train/valid split (train: 75%, valid: 25% of train_valid)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_valid, Y_train_valid, test_size=0.25, random_state=42, stratify=Y_train_valid
    )
    
    # subject 전용 결과 폴더 생성
    subj_folder = os.path.join(RESULT_DIR, f"s{subj.zfill(2)}")
    os.makedirs(subj_folder, exist_ok=True)
    
    # 데이터셋 생성
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(BATCH_SIZE)
    
    # 모델 생성 및 컴파일
    model = build_model(input_shape=(2,4,8,8,1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    
    # callbacks 없이 학습 진행
    history = model.fit(train_dataset, epochs=150, validation_data=val_dataset)
    
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
    
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
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
    
    # 최종 모델 저장 (SavedModel 형식)
    model_save_path = os.path.join(subj_folder, "model_eeg_cnn_transformer.keras")
    model.save(model_save_path)
    print(f"Subject {subj}: Model saved to {model_save_path}")
