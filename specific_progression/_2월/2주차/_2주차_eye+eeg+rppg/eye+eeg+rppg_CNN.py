# CNN 기반의 Dual-Stream Feature Extractor + Cross-Modal Transformer + rPPG 모달 추가
# Stratify 적용, Focal Loss 구현, 오버샘플링 적용

import os
import re
import cv2  # downsample 시 사용
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, Dropout, AveragePooling2D, DepthwiseConv2D, LayerNormalization, 
    MultiHeadAttention, Concatenate, Input, TimeDistributed, LSTM, Add, Softmax, Lambda, 
    MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from collections import Counter

# =============================================================================
# 0. GPU 메모리 제한 (필요 시)
# =============================================================================
def limit_gpu_memory(memory_limit_mib=8000):
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
EPOCHS = 50
LEARNING_RATE = 1e-4

# 📌 데이터 경로
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap_labeled"
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
RPPG_DATA_PATH = "/home/bcml1/yigeon06/PPG_SEG"
SAVE_PATH = "/home/bcml1/sigenv/_2주차_eye+eeg+rppg/result1"
os.makedirs(SAVE_PATH, exist_ok=True)
SUBJECTS = [f"s{str(i).zfill(2)}" for i in range(1, 23)]

# =============================================================================
# 1. 각 모달별 피처 추출기
# -----------------------------------------------------------------------------
def create_eeg_feature_extractor():
    eeg_input = Input(shape=(32,1280,1), name="EEG_Input")
    x = Conv2D(64, kernel_size=(4,16), strides=(2,8), padding='valid', activation='relu')(eeg_input)
    x = DepthwiseConv2D(kernel_size=(6,6), strides=(3,3), padding='valid', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    out = Dense(128, activation='relu')(x)
    return Model(inputs=eeg_input, outputs=out, name="EEGFeatureExtractor")

def create_eye_feature_extractor():
    eye_input = Input(shape=(100,8,64,3), name="Eye_Input")
    x = TimeDistributed(Conv2D(32, kernel_size=(3,3), activation="relu", padding="same"))(eye_input)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(128, activation="relu")(x)
    return Model(inputs=eye_input, outputs=out, name="EyeFeatureExtractor")

def create_rppg_feature_extractor():
    # rPPG 데이터: 입력 shape (100,13,3) (time, roi, rgb)
    rppg_input = Input(shape=(100,13,3), name="RPPG_Input")
    x = Conv2D(16, kernel_size=(2,5), strides=(1,2), padding='valid', activation='relu')(rppg_input)
    x = DepthwiseConv2D(kernel_size=(2,2), padding='valid', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(32, kernel_size=(1,1), padding='valid', activation='relu')(x)
    x = Flatten()(x)
    out = Dense(128, activation='relu')(x)
    return Model(inputs=rppg_input, outputs=out, name="RPPGFeatureExtractor")

# =============================================================================
# 2. Inter-Modality Transformer (두 모달 간 cross-attention)
# -----------------------------------------------------------------------------
def create_inter_modality_fusion(feat1, feat2, num_heads=4, d_model=128, dropout_rate=0.1):
    q = Dense(d_model)(feat1)
    kv = Dense(d_model)(feat2)
    q = Lambda(lambda x: tf.expand_dims(x, axis=1))(q)      # (batch,1,d_model)
    kv = Lambda(lambda x: tf.expand_dims(x, axis=1))(kv)    # (batch,1,d_model)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(query=q, key=kv, value=kv)
    attn = Dropout(dropout_rate)(attn)
    attn = Add()([q, attn])
    attn = LayerNormalization(epsilon=1e-6)(attn)
    attn = Lambda(lambda x: tf.squeeze(x, axis=1))(attn)    # (batch,d_model)
    return attn

# =============================================================================
# 3. Intra-Modality Encoder (각 모달 별 self-attention)
# -----------------------------------------------------------------------------
def create_intra_encoding(feature, num_heads=4, d_model=128, dropout_rate=0.1):
    x = Lambda(lambda x: tf.expand_dims(x, axis=1))(feature)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(query=x, key=x, value=x)
    attn = Dropout(dropout_rate)(attn)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Lambda(lambda x: tf.squeeze(x, axis=1))(x)
    return x

# =============================================================================
# 4. 전체 모델 구성
# -----------------------------------------------------------------------------
def build_full_multimodal_model(num_classes=3):
    # 입력 정의
    eeg_input = Input(shape=(32,1280,1), name="EEG_Input")
    eye_input = Input(shape=(100,8,64,3), name="Eye_Input")
    rppg_input = Input(shape=(100,13,3), name="RPPG_Input")  # rPPG: (time, roi, rgb)
    
    # 피처 추출
    eeg_feat = create_eeg_feature_extractor()(eeg_input)     # (batch,128)
    eye_feat = create_eye_feature_extractor()(eye_input)     # (batch,128)
    rppg_feat = create_rppg_feature_extractor()(rppg_input)    # (batch,128)
    
    # Inter-modality transformers (3 쌍)
    inter_eeg_rppg = create_inter_modality_fusion(eeg_feat, rppg_feat)  # (batch,128)
    inter_eeg_eye  = create_inter_modality_fusion(eeg_feat, eye_feat)
    inter_eye_rppg = create_inter_modality_fusion(eye_feat, rppg_feat)
    
    # Intra-modality encoders (각 모달별)
    intra_eeg  = create_intra_encoding(eeg_feat)
    intra_eye  = create_intra_encoding(eye_feat)
    intra_rppg = create_intra_encoding(rppg_feat)
    
    # 각 branch별 분류기 (총 6 branch)
    pred_inter_eeg_rppg = Dense(num_classes, activation="softmax", name="Inter_EEG_RPPG")(inter_eeg_rppg)
    pred_inter_eeg_eye  = Dense(num_classes, activation="softmax", name="Inter_EEG_Eye")(inter_eeg_eye)
    pred_inter_eye_rppg = Dense(num_classes, activation="softmax", name="Inter_Eye_RPPG")(inter_eye_rppg)
    pred_intra_eeg  = Dense(num_classes, activation="softmax", name="Intra_EEG")(intra_eeg)
    pred_intra_eye  = Dense(num_classes, activation="softmax", name="Intra_Eye")(intra_eye)
    pred_intra_rppg = Dense(num_classes, activation="softmax", name="Intra_RPPG")(intra_rppg)
    
    # 최종 가중치 예측: 6 branch의 raw 피처들을 연결하여 6개 가중치 예측
    fused_for_weight = Concatenate()([inter_eeg_rppg, inter_eeg_eye, inter_eye_rppg,
                                      intra_eeg, intra_eye, intra_rppg])
    weight_logits = Dense(6, activation=None, name="Final_Weight_Logits")(fused_for_weight)
    final_weights = Softmax(name="Final_Weights")(weight_logits)  # (batch,6)
    
    outputs = [pred_inter_eeg_rppg, pred_inter_eeg_eye, pred_inter_eye_rppg,
               pred_intra_eeg, pred_intra_eye, pred_intra_rppg, final_weights]
    
    model = Model(inputs=[eeg_input, eye_input, rppg_input], outputs=outputs, name="Multimodal_Emotion_Classifier")
    return model

# =============================================================================
# 5. 커스텀 모델 클래스 (6 branch 손실을 가중합)
# -----------------------------------------------------------------------------
class FullMultimodalEmotionClassifier(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super(FullMultimodalEmotionClassifier, self).__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data  # y: 정수 레이블 (batch,)
        y_true = y if not isinstance(y, dict) else y["Inter_EEG_RPPG"]
        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            # 7 outputs: 6 branch 예측, 1 final weight
            p1, p2, p3, p4, p5, p6, weights = preds
            loss1 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p1)
            loss2 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p2)
            loss3 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p3)
            loss4 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p4)
            loss5 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p5)
            loss6 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p6)
            losses = tf.stack([loss1, loss2, loss3, loss4, loss5, loss6], axis=-1)  # (batch,6)
            weighted_loss = tf.reduce_sum(weights * losses, axis=-1)
            loss = tf.reduce_mean(weighted_loss)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y_true, p1)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y = data
        y_true = y if not isinstance(y, dict) else y["Inter_EEG_RPPG"]
        preds = self(x, training=False)
        p1 = preds[0]
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, p1))
        self.compiled_metrics.update_state(y_true, p1)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

# =============================================================================
# 6. 클래스별 샘플 오버샘플링 함수
# -----------------------------------------------------------------------------
def oversample_data(train_eeg, train_eye, train_rppg, train_labels):
    unique_classes, counts = np.unique(train_labels, return_counts=True)
    max_count = np.max(counts)
    new_eeg, new_eye, new_rppg, new_labels = [], [], [], []
    for cls in unique_classes:
        indices = np.where(train_labels == cls)[0]
        rep_factor = int(np.ceil(max_count / len(indices)))
        eeg_rep = np.repeat(train_eeg[indices], rep_factor, axis=0)
        eye_rep = np.repeat(train_eye[indices], rep_factor, axis=0)
        rppg_rep = np.repeat(train_rppg[indices], rep_factor, axis=0)
        label_rep = np.repeat(train_labels[indices], rep_factor, axis=0)
        perm = np.random.permutation(eeg_rep.shape[0])
        new_eeg.append(eeg_rep[perm][:max_count])
        new_eye.append(eye_rep[perm][:max_count])
        new_rppg.append(rppg_rep[perm][:max_count])
        new_labels.append(label_rep[perm][:max_count])
    new_eeg = np.concatenate(new_eeg, axis=0)
    new_eye = np.concatenate(new_eye, axis=0)
    new_rppg = np.concatenate(new_rppg, axis=0)
    new_labels = np.concatenate(new_labels, axis=0)
    perm_all = np.random.permutation(new_eeg.shape[0])
    return new_eeg[perm_all], new_eye[perm_all], new_rppg[perm_all], new_labels[perm_all]

# =============================================================================
# 7. rPPG 데이터 로드 함수 (모든 13 ROI를 이어 붙여서)
# -----------------------------------------------------------------------------
def load_rppg_data(subject):
    """
    각 trial에 대해 13 ROI 파일을 이어 붙여 (51,3,13,500) 모양으로 만들고,
    51개 세그먼트에 대해 500프레임 중 균일하게 100프레임을 선택하여
    (51,100,13,3) 모양의 데이터를 얻은 후, 이를 trial 단위로 집계하여
    최종적으로 각 trial에 대해 (51,100,13,3)인 데이터를 반환합니다.
    """
    subject_path = os.path.join(RPPG_DATA_PATH, subject)
    if not os.path.exists(subject_path):
        print(f"🚨 rPPG folder not found for {subject}")
        return None, []
    all_trials = []  # 각 trial의 데이터를 저장 (trial 당 shape: (51,100,13,3))
    sample_indices = []  # trial 번호 기록
    # trial 1~40
    for trial in range(1, 6):
        trial_str = f"{trial:02d}"
        roi_list = []
        for roi in range(13):
            file_name = f"{subject}_trial{trial_str}_roi{roi}.npy"
            file_path = os.path.join(subject_path, file_name)
            if not os.path.exists(file_path):
                print(f"🚨 rPPG file not found: {file_path}")
                continue
            data = np.load(file_path)  # shape: (51,3,1,500)
            roi_list.append(data)  # shape: (51,3,1,500)
        if len(roi_list) != 13:
            print(f"⚠ Trial {trial_str}: expected 13 ROI files, got {len(roi_list)}. Skipping trial.")
            continue
        trial_data = np.concatenate(roi_list, axis=2)  # (51,3,13,500)
        indices = np.linspace(0, 499, num=100, endpoint=True, dtype=int)
        sampled_segments = []
        for seg in range(trial_data.shape[0]):  # 51 segments
            seg_data = trial_data[seg]  # (3,13,500)
            seg_data = seg_data[..., indices]  # (3,13,100)
            seg_data = np.transpose(seg_data, (2,1,0))  # (100,13,3)
            sampled_segments.append(seg_data)
        sampled_segments = np.array(sampled_segments)  # (51,100,13,3)
        all_trials.append(sampled_segments)
        sample_indices.append(trial)
    all_trials = np.array(all_trials)  # (num_trials, 51,100,13,3)
    return all_trials, sample_indices

# =============================================================================
# 8. 기존 EEG, Eye 데이터 로드 함수 (500프레임에서 균일 샘플링으로 100프레임)
# -----------------------------------------------------------------------------
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

# def load_multimodal_data(subject):
#     eeg_data, eye_data, labels, sample_indices = [], [], [], []
#     eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+).npy")
#     for sample_index in range(1, 6):
#         sample_number = f"{sample_index:02d}"
#         print(f"\n🟢 Processing {subject} - Sample {sample_number}")
#         eeg_files = [f for f in os.listdir(EEG_DATA_PATH) if eeg_pattern.match(f) and f"sample_{sample_number}" in f]
#         if not eeg_files:
#             print(f"🚨 No EEG file found for {subject} - Sample {sample_number}")
#             continue
#         for file_name in eeg_files:
#             match = eeg_pattern.match(file_name)
#             if not match:
#                 continue
#             segment_index = int(match.group(2))
#             emotion_label = match.group(3)
#             eeg_file_path = os.path.join(EEG_DATA_PATH, file_name)
#             eeg_segment = np.load(eeg_file_path)  # shape: (32,1280,1)
#             eye_subject_path = os.path.join(EYE_CROP_PATH, subject)
#             if not os.path.exists(eye_subject_path):
#                 print(f"🚨 Subject folder not found: {eye_subject_path}")
#                 continue
#             trial_number = sample_index
#             expected_start = (segment_index - 1) * 500
#             frame_indices = set()
#             file_mapping = {}
#             for f in os.listdir(eye_subject_path):
#                 try:
#                     if not f.startswith(subject) or not f.endswith(".npy"):
#                         continue
#                     match_frame = re.search(r"trial(\d+).avi_frame(\d+)", f)
#                     if not match_frame:
#                         print(f"⚠ Skipping invalid file name: {f} (No trial/frame pattern found)")
#                         continue
#                     file_trial_number = int(match_frame.group(1))
#                     frame_number = int(match_frame.group(2))
#                     if file_trial_number == trial_number:
#                         frame_indices.add(frame_number)
#                         file_mapping[frame_number] = os.path.join(eye_subject_path, f)
#                 except ValueError as e:
#                     print(f"🚨 Error processing file {f}: {e}")
#                     continue
#             frame_indices = sorted(frame_indices)
#             print(f"  🔍 Available frame indices: {frame_indices[:10]} ... {frame_indices[-10:]}")
#             window_length = 500
#             if len(frame_indices) < window_length:
#                 print(f"⚠ Warning: Not enough frames ({len(frame_indices)}) for segment {segment_index:03d}. Skipping Eye Crop.")
#                 eye_data.append(None)
#                 eeg_data.append(eeg_segment)
#                 labels.append(EMOTION_MAPPING[emotion_label])
#                 sample_indices.append(sample_index)
#                 continue
#             selected_frames = frame_indices[:window_length]
#             if len(selected_frames) < window_length:
#                 while len(selected_frames) < window_length:
#                     selected_frames.append(selected_frames[-1])
#                     print("프레임 복제됨")
#             indices = np.linspace(0, window_length - 1, num=100, endpoint=True, dtype=int)
#             selected_frames = [selected_frames[i] for i in indices]
#             eye_frame_files = []
#             for frame in selected_frames:
#                 if frame in file_mapping:
#                     eye_frame_files.append(file_mapping[frame])
#                 if len(eye_frame_files) == 100:
#                     break
#             eye_frame_stack = []
#             for f in eye_frame_files:
#                 frame_data = np.load(f)
#                 frame_data = reshape_eye_frame(frame_data)
#                 if frame_data.shape[-2] == 32:
#                     pad_width = [(0, 0)] * frame_data.ndim
#                     pad_width[-2] = (16, 16)
#                     frame_data = np.pad(frame_data, pad_width, mode='constant', constant_values=0)
#                 eye_frame_stack.append(frame_data)
#             if len(eye_frame_stack) == 100:
#                 eye_data.append(np.stack(eye_frame_stack, axis=0))
#                 eeg_data.append(eeg_segment)
#                 labels.append(EMOTION_MAPPING[emotion_label])
#                 sample_indices.append(sample_index)
#             else:
#                 print(f"⚠ Warning: Found only {len(eye_frame_stack)} matching frames for segment {segment_index:03d}")
#     print(f"✅ Total EEG Samples Loaded: {len(eeg_data)}")
#     print(f"✅ Total Eye Crop Samples Loaded: {len([e for e in eye_data if e is not None])}")
#     print(f"✅ Labels Loaded: {len(labels)}")
#     return (np.array(eeg_data), 
#             np.array([e if e is not None else np.zeros((100,8,64,3)) for e in eye_data]), 
#             np.array(labels), sample_indices)
# 데이터 로드 함수 (500프레임 중 균일 샘플링으로 100프레임 선택)
def load_multimodal_data(subject):
    eeg_data, eye_data, labels, sample_indices = [], [], [], []
    eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+).npy")
    for sample_index in range(1, 6):  
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
# =============================================================================
# 9. 학습 및 평가
# -----------------------------------------------------------------------------
def train_multimodal():
    for subject in SUBJECTS:
        print(f"\n=================== Training subject: {subject} ===================")
        rppg_data, rppg_sample_indices = load_rppg_data(subject)
        if rppg_data is None:
            continue
        # EEG & Eye 데이터 로드
        train_eeg, train_eye, train_labels, test_eeg, test_eye, test_labels = load_multimodal_data(subject)
        # rPPG 데이터 로드 (모든 13 ROI)
        # rppg_data, rppg_sample_indices = load_rppg_data(subject)
        # if rppg_data is None:
        #     continue
        # 주의: EEG/Eye 데이터와 rPPG 데이터의 샘플(세그먼트) 순서는 trial번호 및 segment 순서에 맞게 정렬되어야 함.
        # 여기서는 동일 trial 순서를 가정합니다.
        
        # Train/Validation 분할 (train 데이터를 80:20으로)
        train_eeg, valid_eeg, train_eye, valid_eye, train_labels, valid_labels = train_test_split(
            train_eeg, train_eye, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )
        # rPPG 데이터 분할: rppg_data.shape = (total_segments, 100,13,3)
        split_idx = int(0.8 * rppg_data.shape[0])
        train_rppg = rppg_data[:split_idx]
        valid_rppg = rppg_data[split_idx:]
        
        # 오버샘플링: train 데이터에 대해 (모든 모달)
        train_eeg, train_eye, train_rppg, train_labels = oversample_data(train_eeg, train_eye, train_rppg, train_labels)
        print(f"After oversampling - Train Samples: {train_eeg.shape[0]}")
        
        # 클래스 빈도 기반 alpha 계산
        alpha_values = compute_class_weights(train_labels)
        
        # 모델 생성
        base_model = build_full_multimodal_model(num_classes=3)
        model = FullMultimodalEmotionClassifier(base_model)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=focal_loss(alpha=alpha_values),
                      metrics=["accuracy"])
        model.summary()
        
        for epoch in range(EPOCHS):
            try:
                print(f"\n🚀 Subject {subject} - Epoch {epoch+1}/{EPOCHS}")
                model.fit(
                    [train_eeg, train_eye, train_rppg], train_labels,
                    validation_data=([valid_eeg, valid_eye, valid_rppg], valid_labels),
                    epochs=1, batch_size=BATCH_SIZE
                )
            except tf.errors.ResourceExhaustedError:
                print("⚠️ OOM 발생, 에포크 건너뜁니다.")
                continue
        
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        model.save_weights(weight_path)
        print(f"✅ 모델 가중치 저장됨: {weight_path}")
        
        # 평가 (테스트 데이터는 EEG & Eye는 로드된 test_*, rPPG는 valid 후 남은 부분 사용)
        predictions = model.predict([test_eeg, test_eye, rppg_data[split_idx:]])
        # 예: inter EEG-RPPG branch를 대표로 사용
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
