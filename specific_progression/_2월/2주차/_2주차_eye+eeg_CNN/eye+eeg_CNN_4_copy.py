# CNN 기반의 Dual-Stream Feature Extractor + Cross-Modal Transformer
# Stratify 적용, Focal Loss 구현
#Neutral 학습 및 test가 잘되는지 확인하는 코드

import os
import re
import cv2  # downsample 시 사용
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, Dropout, AveragePooling2D, DepthwiseConv2D, LayerNormalization, 
    MultiHeadAttention, Reshape, Concatenate, GlobalAveragePooling2D, Input, TimeDistributed, 
    LSTM, Add, Dropout, Softmax, Lambda, MaxPooling2D, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from collections import Counter

# =============================================================================
# 0. GPU 메모리 제한 (필요 시)
# =============================================================================
def limit_gpu_memory(memory_limit_mib=8000):
    """Limit TensorFlow GPU memory usage to the specified amount (in MiB)."""
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

limit_gpu_memory(6000)

# 📌 감정 라벨 매핑
EMOTION_MAPPING = {
    "Negative": 0,
    "Positive": 1,
    "Neutral": 2
}

# 📌 하이퍼파라미터 설정
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4

# 📌 데이터 경로
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap_labeled"
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
SAVE_PATH = "/home/bcml1/sigenv/_2주차_eye+eeg_CNN/result4_copy2"
os.makedirs(SAVE_PATH, exist_ok=True)
SUBJECTS = [f"s{str(i).zfill(2)}" for i in range(1, 23)]  # 피실험자 1~22명

# -------------------------
# 📌 Dual-Stream Feature Extractor
def create_dual_stream_feature_extractor():
    """
    EEG와 Eye Crop 데이터를 개별적으로 처리하는 Dual-Stream Feature Extractor
    """
    # 📌 EEG Stream (32채널, 1280 샘플 → (32, 1280, 1))
    eeg_input = Input(shape=(32, 1280, 1), name="EEG_Input")
    x = Conv2D(filters=64, kernel_size=(4, 16), strides=(2, 8), padding='valid', activation='relu')(eeg_input)
    x = DepthwiseConv2D(kernel_size=(6,6), strides=(3,3), padding='valid', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=128, kernel_size=(1,1), strides=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    eeg_output = Dense(128, activation="relu")(x)

    # 📌 Eye Crop Stream (500, 8, 64, 3)
    eye_input = Input(shape=(100, 8, 64, 3), name="Eye_Input")
    eye_cnn = TimeDistributed(Conv2D(32, kernel_size=(3,3), activation="relu", padding="same"))(eye_input)
    eye_cnn = TimeDistributed(MaxPooling2D((2, 2)))(eye_cnn)
    eye_cnn = TimeDistributed(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"))(eye_cnn)
    eye_cnn = TimeDistributed(MaxPooling2D((2, 2)))(eye_cnn)
    eye_cnn = TimeDistributed(Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"))(eye_cnn)
    eye_cnn = TimeDistributed(GlobalAveragePooling2D())(eye_cnn)  # shape: (batch, 100, 128)
    eye_cnn = GlobalAveragePooling1D()(eye_cnn)  # shape: (batch, 128)
    eye_output = Dense(128, activation="relu")(eye_cnn)
    
    model = Model(inputs=[eeg_input, eye_input], outputs=[eeg_output, eye_output], name="DualStreamFeatureExtractor")
    return model

# -------------------------
# 📌 Inter-Modality Fusion Module (Cross-Modal Transformer)
def create_inter_modality_fusion(eeg_features, eye_features, num_heads=4, d_model=128, dropout_rate=0.1):
    """
    EEG와 Eye Crop 간의 관계를 학습하는 Inter-Modality Fusion Module (Cross-Modal Transformer)
    """
    # Cross-Modal Transformer: EEG → Eye Crop
    eeg_query = Dense(d_model)(eeg_features)    # shape: (batch_size, 128)
    eye_key_value = Dense(d_model)(eye_features)
    # Shape 확인
    print("eeg_query shape after Dense:", eeg_query.shape)
    print("eye_key_value shape after Dense:", eye_key_value.shape)
    
    # ✅ 차원 확장 (Lambda 사용) -> 차원확장 제거했었으나, 이렇게 차원확장 하고 아래의 것들도 차원확장하는 것이 오류를 제거할 수 있는 방법이었음.
    eeg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_query)  # (batch_size, 1, 128)
    eye_key_value = Lambda(lambda x: tf.expand_dims(x, axis=1))(eye_key_value)  # (batch_size, 1, 128)
    # 최종 Shape 확인
    print("eeg_query shape after expand_dims:", eeg_query.shape)
    print("eye_key_value shape after expand_dims:", eye_key_value.shape)

    cross_modal_attention_1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_EEG_to_Eye")(
        query=eeg_query, key=eye_key_value, value=eye_key_value
    )
   
    cross_modal_attention_1 = Dropout(dropout_rate)(cross_modal_attention_1)
    cross_modal_attention_1 = Add()([eeg_query, cross_modal_attention_1])
    cross_modal_attention_1 = LayerNormalization(epsilon=1e-6)(cross_modal_attention_1)
    
    cross_modal_attention_1 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_modal_attention_1)  # (batch, d_model)
    
    # Cross-Modal Transformer: Eye Crop → EEG
    eye_query = Dense(d_model)(eye_features)
    eye_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(eye_query)  # Eye → EEG Branch: 시퀀스 길이 1로 확장
    
    eeg_key_value_2 = Dense(d_model)(eeg_features)
    eeg_key_value_2 = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_key_value_2)
    
    cross_modal_attention_2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_Eye_to_EEG")(
        query=eye_query, key=eeg_key_value_2, value=eeg_key_value_2
    )
    cross_modal_attention_2 = Dropout(dropout_rate)(cross_modal_attention_2)
    cross_modal_attention_2 = Add()([eye_query, cross_modal_attention_2])
    cross_modal_attention_2 = LayerNormalization(epsilon=1e-6)(cross_modal_attention_2)

    # 차원 확장 (cross_modal_attention_2를 (None, 128)로 축소)
    cross_modal_attention_2 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_modal_attention_2)  # (batch, d_model)
    
    fused_features = Concatenate(axis=-1)([cross_modal_attention_1, cross_modal_attention_2])
    fused_features = Dense(d_model, activation="relu", name="Fused_Linear")(fused_features)
    
    # Self-Attention Transformer: fused_features를 시퀀스 차원으로 확장하여 Self-Attention 적용 후 squeeze
    fused_features_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(fused_features)  # (batch, 1, d_model)

    # Self-Attention Transformer
    self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttentionFusion")(
        query=fused_features_expanded, key=fused_features_expanded, value=fused_features_expanded
    )
    self_attention = Dropout(dropout_rate)(self_attention)
    self_attention = Add()([fused_features_expanded, self_attention])
    self_attention = LayerNormalization(epsilon=1e-6)(self_attention)

    self_attention = Lambda(lambda x: tf.squeeze(x, axis=1))(self_attention)  # (batch, d_model) -> 차원 축소 추가
    return self_attention

# -------------------------
# Intra-Modality Encoding Module
def create_intra_modality_encoding(eeg_features, eye_features, num_heads=4, d_model=128, dropout_rate=0.1):
    """
    EEG와 Eye Crop의 고유한 특성을 유지하며 강화하는 Intra-Modality Encoding Module
    """
    eeg_features = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_features)  # (batch, 1, 128) -> 차원확장해야함 추가.
    
    eeg_self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_EEG")(
        query=eeg_features, key=eeg_features, value=eeg_features
    )
    eeg_self_attention = Dropout(dropout_rate)(eeg_self_attention)
    eeg_self_attention = Add()([eeg_features, eeg_self_attention])
    eeg_self_attention = LayerNormalization(epsilon=1e-6)(eeg_self_attention)

    eeg_self_attention = Lambda(lambda x: tf.squeeze(x, axis=1))(eeg_self_attention)  # (batch, d_model) -> 차원 축소 추가.
    
    
    # Eye 모달리티: 시퀀스 차원 추가 후 Self-Attention 적용 -> 추가
    eye_features = Lambda(lambda x: tf.expand_dims(x, axis=1))(eye_features)  # (batch, 1, 128)

    eye_self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_Eye")(
        query=eye_features, key=eye_features, value=eye_features
    )
    eye_self_attention = Dropout(dropout_rate)(eye_self_attention)
    eye_self_attention = Add()([eye_features, eye_self_attention])
    eye_self_attention = LayerNormalization(epsilon=1e-6)(eye_self_attention)

    eye_self_attention = Lambda(lambda x: tf.squeeze(x, axis=1))(eye_self_attention)  # (batch, d_model) 차원 축소 추가
    
    return eeg_self_attention, eye_self_attention

# -------------------------
# 원본 Input부터 최종 출력과 커스텀 손실 함수까지 포함하는 단일 모델 생성 함수
def build_combined_model(num_classes=3):
    """
    EEG와 Eye Crop 데이터를 받아 Dual‑Stream Feature Extraction, 
    Cross‑Modal Fusion, Intra‑Modal Encoding을 거쳐 3개의 분류 결과를 출력하는 전체 네트워크를 생성.
    """
    # 1️⃣ 원본 Input 레이어 정의
    eeg_input = Input(shape=(32, 1280, 1), name="EEG_Input")
    eye_input = Input(shape=(100, 8, 64, 3), name="Eye_Input")
    
    # 2️⃣ Dual-Stream Feature Extractor (EEG와 Eye Crop 각각의 특징 추출)
    dual_extractor = create_dual_stream_feature_extractor()
    eeg_features, eye_features = dual_extractor([eeg_input, eye_input])
    
    # 3️⃣ Inter-Modality Fusion Module (Cross-Modal Transformer)
    fused_features = create_inter_modality_fusion(eeg_features, eye_features)

    # 4️⃣ Intra-Modality Encoding Module (각 모달리티의 고유 특성을 강화)
    eeg_encoded, eye_encoded = create_intra_modality_encoding(eeg_features, eye_features)
    
    # 5️⃣ 각 분류 브랜치 구성
    inter_classification = Dense(num_classes, activation="softmax", name="Inter_Classification")(fused_features)
    eeg_classification   = Dense(num_classes, activation="softmax", name="EEG_Classification")(eeg_encoded)
    eye_classification   = Dense(num_classes, activation="softmax", name="EyeCrop_Classification")(eye_encoded)
    
    # 6️⃣ 분류 결과 결합을 위한 가중치(weight) 예측
    concat_features = Concatenate()([fused_features, eeg_encoded, eye_encoded]) # (batch, 384)
    
    # 🔥 1. `Dense(3)`을 확실히 적용
    weights_logits = Dense(units=3, activation=None, name="Weight_Logits")(concat_features)
    
    # 🔥 2. `Softmax`의 `axis`를 명확히 설정(아마 Lambda를 설정하는 것과 성능은 똑같이 나올 것)
    weights = Softmax(axis=-1, name="Weight_Softmax")(weights_logits)
    
    # 🔥 디버깅을 위한 출력
    print("concat_features.shape:", concat_features.shape) #확인
    print("Weight_Logits.shape:", weights_logits.shape) #확인
    print("weights.shape:", weights.shape)  # (batch_size, 3) 예상
    
    # ✅ 최종 모델 생성 (입력: 원본 Input / 출력: 세 개의 분류 결과와 가중치)
    model = Model(inputs=[eeg_input, eye_input],
                  outputs=[inter_classification, eeg_classification, eye_classification, weights],
                  name="Multimodal_Emotion_Classifier")
    
    return model

# -------------------------
# 커스텀 학습 단계를 포함하는 모델 클래스 정의
class MultimodalEmotionClassifier(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super(MultimodalEmotionClassifier, self).__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        # 만약 y가 딕셔너리가 아니면 그대로 사용
        if isinstance(y, dict):
            y_true = y["Inter_Classification"]
        else:
            y_true = y
        with tf.GradientTape() as tape:
            inter_pred, eeg_pred, eye_pred, weights = self(x, training=True)
            loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
            loss_eeg   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eeg_pred)
            loss_eye   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eye_pred)
            w1, w2, w3 = tf.split(weights, num_or_size_splits=3, axis=-1)
            loss = tf.reduce_mean(w1 * loss_inter + w2 * loss_eeg + w3 * loss_eye)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y_true, inter_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results
        # with tf.GradientTape() as tape:
        #     # 모델 예측: outputs는 4개의 텐서로 구성됨
        #     inter_pred, eeg_pred, eye_pred, weights = self(x, training=True)
        #     # 개별 분류 손실 계산 (sparse categorical crossentropy)
        #     loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y, inter_pred)
        #     loss_eeg   = tf.keras.losses.sparse_categorical_crossentropy(y, eeg_pred)
        #     loss_eye   = tf.keras.losses.sparse_categorical_crossentropy(y, eye_pred)
        #     # 예측된 가중치를 분할 (각각 shape: (batch,1))
        #     w1, w2, w3 = tf.split(weights, num_or_size_splits=3, axis=-1)
        #     loss = tf.reduce_mean(w1 * loss_inter + w2 * loss_eeg + w3 * loss_eye)
        # # 그래디언트 계산 및 업데이트
        # trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # # 메트릭 업데이트 (여기서는 inter_pred를 대표 예측값으로 사용)
        # self.compiled_metrics.update_state(y, inter_pred)
        # results = {m.name: m.result() for m in self.metrics}
        # results.update({"loss": loss})
        # return results
    def test_step(self, data):
        x, y = data
        y_true = y["Inter_Classification"] if isinstance(y, dict) else y
        inter_pred, _, _, _ = self(x, training=False)

        loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
        loss = tf.reduce_mean(loss_inter)

        self.compiled_metrics.update_state(y_true, inter_pred)
        metric_results = {m.name: m.result() for m in self.metrics}
        # 만약 "accuracy"가 None이면 0.0을 사용하도록 함.
        acc = metric_results.get("accuracy")
        if acc is None:
            acc = 0.0
        results = {
            "accuracy": acc,
            "loss": loss
        }
        return results


# -------------------------
# 📌 클래스별 샘플 개수를 기반으로 `alpha` 값 자동 계산
def compute_class_weights(labels, num_classes=3):
    """클래스 불균형을 고려하여 `alpha`를 자동 계산하는 함수"""
    class_counts = np.bincount(labels, minlength=num_classes)  # 각 클래스 샘플 수
    total_samples = np.sum(class_counts)  # 전체 샘플 수
    class_weights = total_samples / (num_classes * class_counts)  # 가중치 계산
    class_weights /= np.max(class_weights)  # 정규화 (최대값 기준)
    return class_weights.astype(np.float32)

# -------------------------
# 📌 Focal Loss 정의 (동적 Alpha 적용) 성능을 최적화하고 싶다면 gamma=1.0, 2.0, 3.0 등을 실험하면서 최적의 값을 찾을 수도 있습니다.
def focal_loss(alpha, gamma=2.0):
    """Focal Loss with dynamic alpha"""
    def loss(y_true, y_pred):
        y_true_one_hot = tf.one_hot(y_true, depth=len(alpha))  # 원-핫 변환
        alpha_factor = tf.gather(alpha, tf.argmax(y_true_one_hot, axis=-1))  # 동적 alpha 선택
        loss = -alpha_factor * y_true_one_hot * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred + 1e-7)
        return tf.reduce_mean(loss)
    return loss


# =============================================================================
# 예시: 가중치 분포 로깅 콜백 정의
# =============================================================================
class WeightDistributionLogger(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, batch_size=8):
        super(WeightDistributionLogger, self).__init__()
        self.valid_eeg, self.valid_eye = valid_data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        # 모델 예측: 출력은 [inter_classification, eeg_classification, eye_classification, weights]
        preds = self.model.predict([self.valid_eeg, self.valid_eye], batch_size=self.batch_size)
        weights = preds[3]  # 네 번째 출력이 가중치 값
        mean_weights = np.mean(weights, axis=0)
        std_weights = np.std(weights, axis=0)
        print(f"Epoch {epoch+1}: Weight distribution -> mean: {mean_weights}, std: {std_weights}")

# -------------------------
# 입력 데이터 크기 줄이기 (다운샘플링)
def downsample_eye_frame(frame):
    """Eye Crop 이미지 다운샘플링 (64x32 → 32x16)"""
    return cv2.resize(frame, (32, 8), interpolation=cv2.INTER_AREA)

# Eye Crop 데이터 로드 시 다운샘플링 적용
def reshape_eye_frame(data):
    """
    (N, 32, 64, 3) 형태의 eye frame 데이터를 (32, 64, 3)으로 변환 후 다운샘플링 적용.
    - N이 2 이상이면 평균을 내서 병합.
    - N이 1이면 그대로 사용.
    """
    if len(data.shape) == 4 and data.shape[0] > 0:
        reshaped_data = np.mean(data, axis=0)
        return downsample_eye_frame(reshaped_data)
    elif len(data.shape) == 3 and data.shape == (32, 64, 3):
        return downsample_eye_frame(data)
    else:
        raise ValueError(f"Unexpected shape: {data.shape}, expected (N, 32, 64, 3) or (32, 64, 3)")

# -------------------------
# 데이터 로드 함수 (Sample Index 기준 그룹 단위 8:2 분할)
def load_multimodal_data(subject):
    eeg_data, eye_data, labels, sample_indices = [], [], [], []
    eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+).npy")

    # ✅ Sample 1~40를 대상으로 데이터 로드 (여기서 반복 횟수가 sample의 개수)
    for sample_index in range(1, 41):  
        sample_number = f"{sample_index:02d}"
        print(f"\n🟢 Processing {subject} - Sample {sample_number}")

        eeg_files = [f for f in os.listdir(EEG_DATA_PATH) if eeg_pattern.match(f) and f"sample_{sample_number}" in f]
        if not eeg_files:
            print(f"🚨 No EEG file found for {subject} - Sample {sample_number}")
            continue

        for file_name in eeg_files:
            match = eeg_pattern.match(file_name)
            if not match:
                continue

            segment_index = int(match.group(2))
            emotion_label = match.group(3)
            eeg_file_path = os.path.join(EEG_DATA_PATH, file_name)
            eeg_segment = np.load(eeg_file_path)

            eye_subject_path = os.path.join(EYE_CROP_PATH, subject)
            if not os.path.exists(eye_subject_path):
                print(f"🚨 Subject folder not found: {eye_subject_path}")
                continue

            trial_number = sample_index  # trial_number와 sample_index를 동일하게 사용
            expected_start = (segment_index - 1) * 500
            frame_indices = set()
            file_mapping = {}

            for f in os.listdir(eye_subject_path):
                try:
                    if not f.startswith(subject) or not f.endswith(".npy"):
                        continue
                    match_frame = re.search(r"trial(\d+).avi_frame(\d+)", f)
                    if not match_frame:
                        print(f"⚠ Skipping invalid file name: {f} (No trial/frame pattern found)")
                        continue
                    file_trial_number = int(match_frame.group(1))
                    frame_number = int(match_frame.group(2))
                    if file_trial_number == trial_number:
                        frame_indices.add(frame_number)
                        file_mapping[frame_number] = os.path.join(eye_subject_path, f)
                except ValueError as e:
                    print(f"🚨 Error processing file {f}: {e}")
                    continue

            frame_indices = sorted(frame_indices)
            print(f"  🔍 Available frame indices: {frame_indices[:10]} ... {frame_indices[-10:]}")

            # 원래 500프레임 윈도우 사용
            window_length = 500
            if len(frame_indices) < window_length:
                print(f"⚠ Warning: Not enough frames ({len(frame_indices)}) for segment {segment_index:03d}. Skipping Eye Crop.")
                eye_data.append(None)
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
                sample_indices.append(sample_index)
                continue

            # 500프레임 중 처음 500프레임 선택
            selected_frames = frame_indices[:window_length]

            # 만약 500프레임이 부족한 경우 마지막 프레임 복제 (원래 조건이지만, 여기서는 보통 충분할 것으로 가정)
            if len(selected_frames) < window_length:
                print(f"⚠ Warning: Found only {len(selected_frames)} frames in selected window for segment {segment_index:03d}")
                while len(selected_frames) < window_length:
                    selected_frames.append(selected_frames[-1])
                    print("프레임 복제됨")

            # ★ 균일 샘플링: 500프레임 중 np.linspace를 사용해 100개의 인덱스 선택
            indices = np.linspace(0, window_length - 1, num=100, endpoint=True, dtype=int)
            selected_frames = [selected_frames[i] for i in indices]

            eye_frame_files = []
            for frame in selected_frames:
                if frame in file_mapping:
                    eye_frame_files.append(file_mapping[frame])
                if len(eye_frame_files) == 100:  # 100개로 맞춤
                    break

            eye_frame_stack = []
            for f in eye_frame_files:
                frame_data = np.load(f)
                frame_data = reshape_eye_frame(frame_data)
                if frame_data.shape[-2] == 32:
                    pad_width = [(0, 0)] * frame_data.ndim
                    pad_width[-2] = (16, 16)
                    frame_data = np.pad(frame_data, pad_width, mode='constant', constant_values=0)
                eye_frame_stack.append(frame_data)

            if len(eye_frame_stack) == 100:
                eye_data.append(np.stack(eye_frame_stack, axis=0))
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
                # 각 데이터에 해당하는 sample index 기록
                sample_indices.append(sample_index)
            else:
                print(f"⚠ Warning: Found only {len(eye_frame_stack)} matching frames for segment {segment_index:03d}")

    print(f"✅ Total EEG Samples Loaded: {len(eeg_data)}")
    print(f"✅ Total Eye Crop Samples Loaded: {len([e for e in eye_data if e is not None])}")
    print(f"✅ Labels Loaded: {len(labels)}")
    
    # ===== 그룹 단위 (sample_index)로 train/test 분할 =====
    unique_samples = sorted(set(sample_indices))
    sample_label_dict = {}
    for s, l in zip(sample_indices, labels):
        if s not in sample_label_dict:
            sample_label_dict[s] = l  # 해당 sample의 첫 번째 라벨을 대표로 사용

    unique_labels = [sample_label_dict[s] for s in unique_samples]
    print(f"🔍 Unique Sample Indices: {unique_samples}")
    # # 고유 sample 중 20%를 테스트 그룹으로 분할
    # train_samples, test_samples = train_test_split(unique_samples, test_size=0.2, random_state=42, stratify=unique_labels)
    # print(f"🔍 Train Samples: {train_samples}, Test Samples: {test_samples}")
    # 각 라벨의 샘플 개수를 확인하여 2개 미만인 경우 test에 넣지 않도록 처리
    label_counts = Counter(unique_labels)
    insufficient_classes = {label for label, count in label_counts.items() if count < 2}
    print(f"🔍 Insufficient classes (will be used only in train): {insufficient_classes}")

    # 충분한 샘플이 있는 sample과 그렇지 않은 sample을 분리
    sufficient_samples = [s for s in unique_samples if sample_label_dict[s] not in insufficient_classes]
    sufficient_labels = [sample_label_dict[s] for s in sufficient_samples]
    insufficient_samples = [s for s in unique_samples if sample_label_dict[s] in insufficient_classes]

    # 충분한 샘플들에 대해서 stratify를 적용한 train/test split 수행
    if sufficient_samples:
        train_suff, test_suff = train_test_split(
            sufficient_samples, 
            test_size=0.2, 
            random_state=42, 
            stratify=sufficient_labels
        )
    else:
        train_suff, test_suff = [], []
    train_samples = train_suff + insufficient_samples
    test_samples = test_suff
    # 각 데이터가 속한 sample index를 기준으로 분할
    train_eeg_data, train_eye_data, train_labels = [], [], []
    test_eeg_data, test_eye_data, test_labels = [], [], []

    for idx, s in enumerate(sample_indices):
        if s in train_samples:
            train_eeg_data.append(eeg_data[idx])
            train_eye_data.append(eye_data[idx])
            train_labels.append(labels[idx])
        else:
            test_eeg_data.append(eeg_data[idx])
            test_eye_data.append(eye_data[idx])
            test_labels.append(labels[idx])

    # Numpy 배열로 변환 (eye 데이터는 None인 경우 0으로 채운 배열로 대체)
    train_eeg_data = np.array(train_eeg_data)
    train_eye_data = np.array([e if e is not None else np.zeros((100, 8, 64, 3)) for e in train_eye_data])
    train_labels = np.array(train_labels)
    
    test_eeg_data = np.array(test_eeg_data)
    test_eye_data = np.array([e if e is not None else np.zeros((100, 8, 64, 3)) for e in test_eye_data])
    test_labels = np.array(test_labels)
    
    print(f"✅ Train EEG Samples: {train_eeg_data.shape[0]}, Test EEG Samples: {test_eeg_data.shape[0]}")
    print(f"✅ Train Eye Crop Samples: {train_eye_data.shape[0]}, Test Eye Crop Samples: {test_eye_data.shape[0]}")
    print(f"✅ Train Labels: {train_labels.shape[0]}, Test Labels: {test_labels.shape[0]}")
    
    return train_eeg_data, train_eye_data, train_labels, test_eeg_data, test_eye_data, test_labels

# -------------------------
# 학습 및 평가
def train_multimodal():
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    for subject in subjects:
        print(f"Training subject: {subject}")

        # ✅ 수정: load_multimodal_data()에서 이미 8:2로 나눠진 데이터를 가져옴.
        train_eeg, train_eye, train_labels, test_eeg, test_eye, test_labels = load_multimodal_data(subject)

        # ✅ 추가: Train 데이터를 80:20으로 Validation Set으로 분할
        train_eeg, valid_eeg, train_eye, valid_eye, train_labels, valid_labels = train_test_split(
            train_eeg, train_eye, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )

        # ✅ 클래스 빈도수 기반 `alpha` 계산
        alpha_values = compute_class_weights(train_labels)
        
        # ✅ 기존 모델 생성 후, MultimodalEmotionClassifier 래핑
        # 모델 생성 및 래핑
        base_model = build_combined_model(num_classes=3)
        model = MultimodalEmotionClassifier(base_model)
        # 커스텀 train_step과 test_step을 오버라이드했으므로 loss 인자는 생략해도 됩니다.
         # ============================
        # Option 1: Focal Loss 사용 (기존 설정)
        # ============================
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        #               loss=focal_loss(alpha=alpha_values),
        #               metrics=["accuracy"])
        # ============================
        # Option 2: Standard Cross-Entropy Loss 사용
        # (Neutral 클래스 인식을 확인하기 위해 주석 해제 후 사용)
        # ============================
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"])
        
        print(model.summary())
        # WeightDistributionLogger 콜백 생성 (검증 데이터 전달)
        weight_logger = WeightDistributionLogger(valid_data=(valid_eeg, valid_eye), batch_size=BATCH_SIZE)
        
        start_epoch = 0
        max_epochs = EPOCHS
        batch_size = BATCH_SIZE
        max_retries = 3

        for epoch in range(start_epoch, max_epochs):
            retries = 0
            while retries < max_retries:
                try:
                    print(f"\n🚀 Training {subject} - Epoch {epoch+1}/{max_epochs} (Retry: {retries})...")
                    
                    dummy_y_train = {
                        "Inter_Classification": train_labels,
                        "EEG_Classification": train_labels,
                        "EyeCrop_Classification": train_labels,
                        "Weight_Softmax": train_labels
                    }

                    dummy_y_valid = {
                        "Inter_Classification": valid_labels,
                        "EEG_Classification": valid_labels,
                        "EyeCrop_Classification": valid_labels,
                        "Weight_Softmax": valid_labels
                    }

                    # model.fit(
                    #     [train_eeg, train_eye], train_labels,
                    #     #dummy_y_train,
                    #     validation_data=([valid_eeg, valid_eye], valid_labels),
                    #     epochs=1,
                    #     batch_size=batch_size
                    # )
                    
                    model.fit(
                        [train_eeg, train_eye], train_labels,
                        validation_data=([valid_eeg, valid_eye], valid_labels),
                        epochs=1,
                        batch_size=batch_size,
                        callbacks=[weight_logger]  # 여기에 WeightDistributionLogger 적용
                    )
                    
                    # model.fit(
                    #     [train_eeg, train_eye], train_labels,
                    #     validation_data=([valid_eeg, valid_eye], valid_labels),
                    #     epochs=1, batch_size=batch_size
                  # )
                    
                    break  
                except tf.errors.ResourceExhaustedError:
                    print(f"⚠️ OOM 발생! 재시작 (Retry: {retries+1})...")
                    tf.keras.backend.clear_session()
                    import gc
                    gc.collect()
                    model = build_combined_model(num_classes=3)
                    retries += 1
                    tf.keras.backend.sleep(1)
            else:
                print(f"❌ 에포크 {epoch+1}에서 최대 재시도 횟수를 초과하였습니다. 다음 subject로 넘어갑니다.")
                break

        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        model.save_weights(weight_path)
        print(f"✅ 모델 가중치 저장됨: {weight_path}")

        # ✅ 예측 결과 처리
        predictions = model.predict([test_eeg, test_eye])
        inter_pred = predictions[0]  # 평가에 사용할 분류 브랜치 선택
        predicted_labels = np.argmax(inter_pred, axis=-1)

        test_report = classification_report(
            test_labels, predicted_labels,
            target_names=["Negative", "Positive", "Neutral"],
            labels=[0, 1, 2],
            zero_division=0
        )
        print(f"\n📊 Test Report for {subject}")
        print(test_report)

        report_path = os.path.join(subject_save_path, f"{subject}_test_report.txt")
        with open(report_path, "w") as f:
            f.write(test_report)
        print(f"✅ 테스트 리포트 저장됨: {report_path}")
        
if __name__ == "__main__":
    train_multimodal()
  