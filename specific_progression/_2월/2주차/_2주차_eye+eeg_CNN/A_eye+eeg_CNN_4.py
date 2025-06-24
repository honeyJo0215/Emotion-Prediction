# CNN 기반의 Dual-Stream Feature Extractor + Cross-Modal Transformer
# Stratify 적용, Focal Loss 구현

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # GPU 메모리 자동 증가 방지
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow 로그 최소화
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

limit_gpu_memory(8000)


# 📌 감정 라벨 매핑
EMOTION_MAPPING = {
    "Negative": 0,
    "Positive": 1,
    "Neutral": 2
}

# 📌 하이퍼파라미터 설정
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4

# 📌 데이터 경로
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap_labeled"
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
SAVE_PATH = "/home/bcml1/sigenv/_2주차_eye+eeg_CNN/A_result5"
os.makedirs(SAVE_PATH, exist_ok=True)
SUBJECTS = [f"s{str(i).zfill(2)}" for i in range(14, 23)]  # 피실험자 1~22명

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

    # 📌 Eye Crop Stream (원본은 500프레임, 여기서는 균일 샘플링으로 100프레임 사용)
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
    eeg_query = Dense(d_model)(eeg_features)    # shape: (batch, 128)
    eye_key_value = Dense(d_model)(eye_features)
    print("eeg_query shape after Dense:", eeg_query.shape)
    print("eye_key_value shape after Dense:", eye_key_value.shape)
    
    eeg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_query)  # (batch, 1, 128)
    eye_key_value = Lambda(lambda x: tf.expand_dims(x, axis=1))(eye_key_value)  # (batch, 1, 128)
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
    eye_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(eye_query)
    eeg_key_value_2 = Dense(d_model)(eeg_features)
    eeg_key_value_2 = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_key_value_2)
    
    cross_modal_attention_2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_Eye_to_EEG")(
        query=eye_query, key=eeg_key_value_2, value=eeg_key_value_2
    )
    cross_modal_attention_2 = Dropout(dropout_rate)(cross_modal_attention_2)
    cross_modal_attention_2 = Add()([eye_query, cross_modal_attention_2])
    cross_modal_attention_2 = LayerNormalization(epsilon=1e-6)(cross_modal_attention_2)
    cross_modal_attention_2 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_modal_attention_2)  # (batch, d_model)
    
    fused_features = Concatenate(axis=-1)([cross_modal_attention_1, cross_modal_attention_2])
    fused_features = Dense(d_model, activation="relu", name="Fused_Linear")(fused_features)
    
    fused_features_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(fused_features)  # (batch, 1, d_model)
    self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttentionFusion")(
        query=fused_features_expanded, key=fused_features_expanded, value=fused_features_expanded
    )
    self_attention = Dropout(dropout_rate)(self_attention)
    self_attention = Add()([fused_features_expanded, self_attention])
    self_attention = LayerNormalization(epsilon=1e-6)(self_attention)
    self_attention = Lambda(lambda x: tf.squeeze(x, axis=1))(self_attention)  # (batch, d_model)
    return self_attention

# -------------------------
# Intra-Modality Encoding Module
def create_intra_modality_encoding(eeg_features, eye_features, num_heads=4, d_model=128, dropout_rate=0.1):
    """
    EEG와 Eye Crop의 고유한 특성을 유지하며 강화하는 Intra-Modality Encoding Module
    """
    eeg_features = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_features)  # (batch, 1, 128)
    eeg_self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_EEG")(
        query=eeg_features, key=eeg_features, value=eeg_features
    )
    eeg_self_attention = Dropout(dropout_rate)(eeg_self_attention)
    eeg_self_attention = Add()([eeg_features, eeg_self_attention])
    eeg_self_attention = LayerNormalization(epsilon=1e-6)(eeg_self_attention)
    eeg_self_attention = Lambda(lambda x: tf.squeeze(x, axis=1))(eeg_self_attention)  # (batch, d_model)
    
    eye_features = Lambda(lambda x: tf.expand_dims(x, axis=1))(eye_features)  # (batch, 1, 128)
    eye_self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_Eye")(
        query=eye_features, key=eye_features, value=eye_features
    )
    eye_self_attention = Dropout(dropout_rate)(eye_self_attention)
    eye_self_attention = Add()([eye_features, eye_self_attention])
    eye_self_attention = LayerNormalization(epsilon=1e-6)(eye_self_attention)
    eye_self_attention = Lambda(lambda x: tf.squeeze(x, axis=1))(eye_self_attention)  # (batch, d_model)
    
    return eeg_self_attention, eye_self_attention

# -------------------------
# 원본 Input부터 최종 출력과 커스텀 손실 함수까지 포함하는 단일 모델 생성 함수
def build_combined_model(num_classes=3):
    """
    EEG와 Eye Crop 데이터를 받아 Dual‑Stream Feature Extraction, 
    Cross‑Modal Fusion, Intra‑Modal Encoding을 거쳐 3개의 분류 결과를 출력하는 전체 네트워크 생성.
    """
    eeg_input = Input(shape=(32, 1280, 1), name="EEG_Input")
    eye_input = Input(shape=(100, 8, 64, 3), name="Eye_Input")
    
    dual_extractor = create_dual_stream_feature_extractor()
    eeg_features, eye_features = dual_extractor([eeg_input, eye_input])
    
    fused_features = create_inter_modality_fusion(eeg_features, eye_features)
    eeg_encoded, eye_encoded = create_intra_modality_encoding(eeg_features, eye_features)
    
    inter_classification = Dense(num_classes, activation="softmax", name="Inter_Classification")(fused_features)
    eeg_classification   = Dense(num_classes, activation="softmax", name="EEG_Classification")(eeg_encoded)
    eye_classification   = Dense(num_classes, activation="softmax", name="EyeCrop_Classification")(eye_encoded)
    
    concat_features = Concatenate()([fused_features, eeg_encoded, eye_encoded])
    weights_logits = Dense(units=3, activation=None, name="Weight_Logits")(concat_features)
    weights = Softmax(axis=-1, name="Weight_Softmax")(weights_logits)
    
    print("concat_features.shape:", concat_features.shape)
    print("Weight_Logits.shape:", weights_logits.shape)
    print("weights.shape:", weights.shape)
    
    model = Model(inputs=[eeg_input, eye_input],
                  outputs=[inter_classification, eeg_classification, eye_classification, weights],
                  name="Multimodal_Emotion_Classifier")
    return model

# -------------------------
# 커스텀 학습 단계를 포함하는 모델 클래스 정의
# class MultimodalEmotionClassifier(tf.keras.Model):
#     def __init__(self, base_model, **kwargs):
#         super(MultimodalEmotionClassifier, self).__init__(**kwargs)
#         self.base_model = base_model

#     def call(self, inputs, training=False):
#         return self.base_model(inputs, training=training)

#     def train_step(self, data):
#         x, y = data
#         if isinstance(y, dict):
#             y_true = y["Inter_Classification"]
#         else:
#             y_true = y
#         with tf.GradientTape() as tape:
#             inter_pred, eeg_pred, eye_pred, weights = self(x, training=True)
#             loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
#             loss_eeg   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eeg_pred)
#             loss_eye   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eye_pred)
#             w1, w2, w3 = tf.split(weights, num_or_size_splits=3, axis=-1)
#             loss = tf.reduce_mean(w1 * loss_inter + w2 * loss_eeg + w3 * loss_eye)
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         self.compiled_metrics.update_state(y_true, inter_pred)
#         results = {m.name: m.result() for m in self.metrics}
#         results.update({"loss": loss})
#         return results

#     def test_step(self, data):
#         x, y = data
#         y_true = y["Inter_Classification"] if isinstance(y, dict) else y
#         inter_pred, _, _, _ = self(x, training=False)
#         loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
#         loss = tf.reduce_mean(loss_inter)
#         self.compiled_metrics.update_state(y_true, inter_pred)
#         metric_results = {m.name: m.result() for m in self.metrics}
#         acc = metric_results.get("accuracy")
#         if acc is None:
#             acc = 0.0
#         results = {"accuracy": acc, "loss": loss}
#         return results

#(수정 버전)
class MultimodalEmotionClassifier(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super(MultimodalEmotionClassifier, self).__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        # y가 dict 형태라면 Inter_Classification 키를 사용
        if isinstance(y, dict):
            y_true = y["Inter_Classification"]
        else:
            y_true = y

        with tf.GradientTape() as tape:
            # 모델은 네 개의 출력을 내보내지만, 여기서는 동적 가중치 branch는 loss 계산에 사용하지 않고 무시합니다.
            inter_pred, eeg_pred, eye_pred, _ = self(x, training=True)
            loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
            loss_eeg   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eeg_pred)
            loss_eye   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eye_pred)
            # 단순 평균하여 전체 loss로 사용
            loss = tf.reduce_mean((loss_inter + loss_eeg + loss_eye) / 3.0)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y_true, inter_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y = data
        y_true = y["Inter_Classification"] if isinstance(y, dict) else y
        inter_pred, _, _, _ = self(x, training=False)
        loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
        loss = tf.reduce_mean(loss_inter)
        self.compiled_metrics.update_state(y_true, inter_pred)
        metric_results = {m.name: m.result() for m in self.metrics}
        results = {"accuracy": metric_results.get("accuracy", 0.0), "loss": loss}
        return results


# -------------------------
# 클래스별 샘플 개수를 기반으로 `alpha` 값 자동 계산
def compute_class_weights(labels, num_classes=3):
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights /= np.max(class_weights)
    return class_weights.astype(np.float32)

# -------------------------
# Focal Loss 정의 (동적 Alpha 적용)
def focal_loss(alpha, gamma=2.0):
    def loss(y_true, y_pred):
        y_true_one_hot = tf.one_hot(y_true, depth=len(alpha))
        alpha_factor = tf.gather(alpha, tf.argmax(y_true_one_hot, axis=-1))
        loss_val = -alpha_factor * y_true_one_hot * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred + 1e-7)
        return tf.reduce_mean(loss_val)
    return loss

# -------------------------
# 입력 데이터 크기 줄이기 (다운샘플링)
def downsample_eye_frame(frame):
    return cv2.resize(frame, (32, 8), interpolation=cv2.INTER_AREA)

def reshape_eye_frame(data):
    if len(data.shape) == 4 and data.shape[0] > 0:
        reshaped_data = np.mean(data, axis=0)
        return downsample_eye_frame(reshaped_data)
    elif len(data.shape) == 3 and data.shape == (32, 64, 3):
        return downsample_eye_frame(data)
    else:
        raise ValueError(f"Unexpected shape: {data.shape}, expected (N, 32, 64, 3) or (32, 64, 3)")

# -------------------------
# 데이터 로드 함수 (500프레임 중 균일 샘플링으로 100프레임 선택)
def load_multimodal_data(subject):
    eeg_data, eye_data, labels, sample_indices = [], [], [], []
    eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+).npy")
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
            trial_number = sample_index  
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
            if len(frame_indices) > 0:
                print(f"  🔍 Available frame indices: {frame_indices[:10]} ... {frame_indices[-10:]}")
            else:
                print("  🔍 No frame indices found")
            window_length = 500
            if len(frame_indices) < window_length:
                print(f"⚠ Warning: Not enough frames ({len(frame_indices)}) for segment {segment_index:03d}. Skipping Eye Crop.")
                eye_data.append(None)
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
                sample_indices.append(sample_index)
                continue
            selected_frames = frame_indices[:window_length]
            if len(selected_frames) < window_length:
                print(f"⚠ Warning: Found only {len(selected_frames)} frames in selected window for segment {segment_index:03d}")
                while len(selected_frames) < window_length:
                    selected_frames.append(selected_frames[-1])
                    print("프레임 복제됨")
            indices = np.linspace(0, window_length - 1, num=100, endpoint=True, dtype=int)
            selected_frames = [selected_frames[i] for i in indices]
            eye_frame_files = []
            for frame in selected_frames:
                if frame in file_mapping:
                    eye_frame_files.append(file_mapping[frame])
                if len(eye_frame_files) == 100:
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
                sample_indices.append(sample_index)
            else:
                print(f"⚠ Warning: Found only {len(eye_frame_stack)} matching frames for segment {segment_index:03d}")
    print(f"✅ Total EEG Samples Loaded: {len(eeg_data)}")
    print(f"✅ Total Eye Crop Samples Loaded: {len([e for e in eye_data if e is not None])}")
    print(f"✅ Labels Loaded: {len(labels)}")
    
    # 데이터를 numpy 배열로 변환 (Eye 데이터가 None인 경우 제로 배열로 대체)
    eeg_data = np.array(eeg_data)
    eye_data = np.array([e if e is not None else np.zeros((100, 8, 64, 3)) for e in eye_data])
    labels = np.array(labels)
    
    # sklearn의 train_test_split을 사용하여 train과 test 데이터셋으로 분리 (80%:20%, stratify 적용)
    from sklearn.model_selection import train_test_split
    train_eeg, test_eeg, train_eye, test_eye, train_labels, test_labels = train_test_split(
        eeg_data, eye_data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return train_eeg, train_eye, train_labels, test_eeg, test_eye, test_labels

# -------------------------
# 오버샘플링 함수: 부족한 클래스의 데이터를 복제하여 최대 샘플 수와 맞춤.
def oversample_data(train_eeg, train_eye, train_labels):
    unique_classes, counts = np.unique(train_labels, return_counts=True)
    max_count = np.max(counts)
    new_train_eeg = []
    new_train_eye = []
    new_train_labels = []
    for cls in unique_classes:
        indices = np.where(train_labels == cls)[0]
        n_samples = len(indices)
        replication_factor = int(np.ceil(max_count / n_samples))
        eeg_data_cls = train_eeg[indices]
        eye_data_cls = train_eye[indices]
        label_data_cls = train_labels[indices]
        eeg_rep = np.repeat(eeg_data_cls, replication_factor, axis=0)
        eye_rep = np.repeat(eye_data_cls, replication_factor, axis=0)
        label_rep = np.repeat(label_data_cls, replication_factor, axis=0)
        perm = np.random.permutation(eeg_rep.shape[0])
        eeg_rep = eeg_rep[perm][:max_count]
        eye_rep = eye_rep[perm][:max_count]
        label_rep = label_rep[perm][:max_count]
        new_train_eeg.append(eeg_rep)
        new_train_eye.append(eye_rep)
        new_train_labels.append(label_rep)
    new_train_eeg = np.concatenate(new_train_eeg, axis=0)
    new_train_eye = np.concatenate(new_train_eye, axis=0)
    new_train_labels = np.concatenate(new_train_labels, axis=0)
    perm_all = np.random.permutation(new_train_eeg.shape[0])
    new_train_eeg = new_train_eeg[perm_all]
    new_train_eye = new_train_eye[perm_all]
    new_train_labels = new_train_labels[perm_all]
    return new_train_eeg, new_train_eye, new_train_labels

# -------------------------
# 학습 및 평가
def train_multimodal():
    subjects = [f"s{str(i).zfill(2)}" for i in range(14, 23)]
    for subject in subjects:
        print(f"Training subject: {subject}")
        train_eeg, train_eye, train_labels, test_eeg, test_eye, test_labels = load_multimodal_data(subject)
        
        # Train 데이터를 80:20으로 Validation Set으로 분할 (stratify 적용)
        train_eeg, valid_eeg, train_eye, valid_eye, train_labels, valid_labels = train_test_split(
            train_eeg, train_eye, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )
        
        # 오버샘플링 적용: train set에서 각 클래스 샘플 수를 최대 샘플 수로 맞춤.
        train_eeg, train_eye, train_labels = oversample_data(train_eeg, train_eye, train_labels)
        print(f"After oversampling - Train EEG Samples: {train_eeg.shape[0]}, Train Labels: {train_labels.shape[0]}")
        
        # 클래스 빈도수 기반 alpha 계산 (oversampling 후에는 거의 균등하지만, 혹시 모르니 계산)
        alpha_values = compute_class_weights(train_labels)
        
        # 모델 생성 및 커스텀 모델 클래스 래핑
        base_model = build_combined_model(num_classes=3)
        model = MultimodalEmotionClassifier(base_model)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=focal_loss(alpha=alpha_values),
                      metrics=["accuracy"])
        model.summary()

        start_epoch = 0
        max_epochs = EPOCHS
        batch_size = BATCH_SIZE
        max_retries = 3

        for epoch in range(start_epoch, max_epochs):
            retries = 0
            while retries < max_retries:
                try:
                    print(f"\n🚀 Training {subject} - Epoch {epoch+1}/{max_epochs} (Retry: {retries})...")
                    model.fit(
                        [train_eeg, train_eye], train_labels,
                        validation_data=([valid_eeg, valid_eye], valid_labels),
                        epochs=1,
                        batch_size=batch_size
                    )
                    break  
                except tf.errors.ResourceExhaustedError:
                    print(f"⚠️ OOM 발생! 재시작 (Retry: {retries+1})...")
                    tf.keras.backend.clear_session()
                    import gc
                    gc.collect()
                    base_model = build_combined_model(num_classes=3)
                    model = MultimodalEmotionClassifier(base_model)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                                  loss=focal_loss(alpha=alpha_values),
                                  metrics=["accuracy"])
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

        predictions = model.predict([test_eeg, test_eye])
        inter_pred = predictions[0]
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
