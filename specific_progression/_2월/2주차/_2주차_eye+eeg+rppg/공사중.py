import os
import re
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, Dropout, AveragePooling2D, DepthwiseConv2D, LayerNormalization,
    BatchNormalization, MultiHeadAttention, Concatenate, Input, TimeDistributed, Add, Softmax, Lambda,
    MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

# =============================================================================
# 0. GPU 메모리 제한
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
EMOTION_MAPPING = {"Negative": 0, "Positive": 1, "Neutral": 2}

# 📌 하이퍼파라미터 설정
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-3  # 약간 높은 초기 학습률

# 📌 데이터 경로
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap_labeled"
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
RPPG_DATA_PATH = "/home/bcml1/yigeon06/PPG_SEG"  # rPPG 파일들이 저장된 경로
SAVE_PATH = "/home/bcml1/sigenv/_2주차_eye+eeg+rppg/result1"
os.makedirs(SAVE_PATH, exist_ok=True)
SUBJECTS = [f"s{str(i).zfill(2)}" for i in range(1, 23)]

# =============================================================================
# 1. 각 모달별 피처 추출기
# -----------------------------------------------------------------------------
def create_eeg_feature_extractor():
    eeg_input = Input(shape=(32,1280,1), name="EEG_Input")
    x = Conv2D(64, kernel_size=(4,16), strides=(2,8), padding='valid', activation='relu')(eeg_input)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(6,6), strides=(3,3), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    out = Dense(128, activation='relu')(x)
    out = Dropout(0.3)(out)
    return Model(inputs=eeg_input, outputs=out, name="EEGFeatureExtractor")

def create_eye_feature_extractor():
    eye_input = Input(shape=(100,8,64,3), name="Eye_Input")
    x = TimeDistributed(Conv2D(32, kernel_size=(3,3), activation="relu", padding="same"))(eye_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(128, activation="relu")(x)
    out = Dropout(0.3)(out)
    return Model(inputs=eye_input, outputs=out, name="EyeFeatureExtractor")

def build_segment_cnn():
    seg_input = Input(shape=(100,13,3))
    y = Conv2D(16, kernel_size=(2,5), strides=(1,2), padding='valid', activation='relu')(seg_input)
    y = BatchNormalization()(y)
    y = DepthwiseConv2D(kernel_size=(2,2), padding='valid', activation='relu')(y)
    y = BatchNormalization()(y)
    y = AveragePooling2D(pool_size=(2,2), strides=(2,2))(y)
    y = Conv2D(32, kernel_size=(1,1), padding='valid', activation='relu')(y)
    y = BatchNormalization()(y)
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dropout(0.3)(y)
    return Model(inputs=seg_input, outputs=y, name="SegmentCNN")

def create_rppg_feature_extractor():
    rppg_input = Input(shape=(100,13,3), name="RPPG_Input")
    segment_model = build_segment_cnn()
    x = segment_model(rppg_input)
    return Model(inputs=rppg_input, outputs=x, name="RPPGFeatureExtractor")

# =============================================================================
# 2. Inter-Modality Transformer (두 모달 간 cross-attention)
# -----------------------------------------------------------------------------
def create_inter_modality_fusion(feat1, feat2, num_heads=4, d_model=128, dropout_rate=0.1):
    q = Dense(d_model)(feat1)
    kv = Dense(d_model)(feat2)
    q = Lambda(lambda x: tf.expand_dims(x, axis=1))(q)
    kv = Lambda(lambda x: tf.expand_dims(x, axis=1))(kv)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(query=q, key=kv, value=kv)
    attn = Dropout(dropout_rate)(attn)
    attn = Add()([q, attn])
    attn = LayerNormalization(epsilon=1e-6)(attn)
    attn = Lambda(lambda x: tf.squeeze(x, axis=1))(attn)
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
# 4. 전체 모델 구성 (출력을 dict로 반환하여 각 branch에 이름 부여)
# -----------------------------------------------------------------------------
def build_full_multimodal_model(num_classes=3):
    # 입력 정의
    eeg_input = Input(shape=(32,1280,1), name="EEG_Input")
    eye_input = Input(shape=(100,8,64,3), name="Eye_Input")
    rppg_input = Input(shape=(100,13,3), name="RPPG_Input")
    
    # 피처 추출
    eeg_feat = create_eeg_feature_extractor()(eeg_input)
    eye_feat = create_eye_feature_extractor()(eye_input)
    rppg_feat = create_rppg_feature_extractor()(rppg_input)
    
    # Inter-modality transformers (3 쌍)
    inter_eeg_rppg = create_inter_modality_fusion(eeg_feat, rppg_feat)
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
    
    # 최종 가중치 예측 모듈: 6 branch의 raw 피처들을 연결하여 6개 가중치 예측
    fused_for_weight = Concatenate()([inter_eeg_rppg, inter_eeg_eye, inter_eye_rppg,
                                      intra_eeg, intra_eye, intra_rppg])
    weight_logits = Dense(6, activation=None, name="Final_Weight_Logits")(fused_for_weight)
    final_weights = Softmax(name="Final_Weights")(weight_logits)
    
    outputs = {
        "Inter_EEG_RPPG": pred_inter_eeg_rppg,
        "Inter_EEG_Eye": pred_inter_eeg_eye,
        "Inter_Eye_RPPG": pred_inter_eye_rppg,
        "Intra_EEG": pred_intra_eeg,
        "Intra_Eye": pred_intra_eye,
        "Intra_RPPG": pred_intra_rppg,
        "Final_Weights": final_weights
    }
    
    model = Model(inputs=[eeg_input, eye_input, rppg_input], outputs=outputs, name="Multimodal_Emotion_Classifier")
    return model

# =============================================================================
# 5. 클래스별 오버샘플링 함수
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
# 6. 모든 모달 로드 함수: EEG, Eye, rPPG 데이터를 trial 및 segment 단위로 로드
# -----------------------------------------------------------------------------
def load_rppg_data(subject):
    subject_path = os.path.join(RPPG_DATA_PATH, subject)
    if not os.path.exists(subject_path):
        print(f"🚨 rPPG folder not found for {subject}")
        return None, []
    all_segments = []
    sample_indices = []
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
            roi_list.append(data)
        if len(roi_list) != 13:
            print(f"⚠ Trial {trial_str}: expected 13 ROI files, got {len(roi_list)}. Skipping trial.")
            continue
        trial_data = np.concatenate(roi_list, axis=2)  # (51,3,13,500)
        indices = np.linspace(0, 499, num=100, endpoint=True, dtype=int)
        for seg in range(trial_data.shape[0]):  # 51 segments
            seg_data = trial_data[seg]  # (3,13,500)
            seg_data = seg_data[..., indices]  # (3,13,100)
            seg_data = np.transpose(seg_data, (2,1,0))  # (100,13,3)
            all_segments.append(seg_data)
            sample_indices.append(trial)
    all_segments = np.array(all_segments)
    return all_segments, sample_indices

# -----------------------------------------------------------------------------
# EEG & Eye 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
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
            eeg_segment = np.load(eeg_file_path)  # shape: (32,1280,1)
            eye_subject_path = os.path.join(EYE_CROP_PATH, subject)
            if not os.path.exists(eye_subject_path):
                print(f"🚨 Subject folder not found: {eye_subject_path}")
                continue
            trial_number = sample_index
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
    return (np.array(eeg_data), 
            np.array([e if e is not None else np.zeros((100,8,64,3)) for e in eye_data]), 
            np.array(labels), sample_indices)

def load_all_modalities(subject):
    eeg_data, eye_data, labels, seg_ids = load_multimodal_data(subject)
    rppg_data, rppg_seg_ids = load_rppg_data(subject)
    if rppg_data is None:
        return eeg_data, eye_data, None, labels, seg_ids
    if len(labels) != len(rppg_data):
        print(f"Warning: EEG/Eye segments: {len(labels)}, rPPG segments: {len(rppg_data)}")
    else:
        eeg_order = sorted(range(len(seg_ids)), key=lambda i: seg_ids[i])
        rppg_order = sorted(range(len(rppg_seg_ids)), key=lambda i: rppg_seg_ids[i])
        eeg_data = eeg_data[eeg_order]
        eye_data = eye_data[eeg_order]
        labels = labels[eeg_order]
        rppg_data = rppg_data[rppg_order]
        seg_ids = [seg_ids[i] for i in eeg_order]
    eeg_data = eeg_data.astype(np.float32)
    eye_data = eye_data.astype(np.float32) / 255.0
    rppg_data = rppg_data.astype(np.float32) / 255.0
    return eeg_data, eye_data, rppg_data, np.array(labels), seg_ids

# =============================================================================
# 7. Focal Loss 및 클래스 가중치 계산
# -----------------------------------------------------------------------------
def compute_class_weights(labels, num_classes=3):
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights /= np.max(class_weights)
    return class_weights.astype(np.float32)

def focal_loss(alpha, gamma=2.0):
    """
    각 배치 샘플별 손실을 반환하도록 수정 (shape: [batch_size])
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=len(alpha))
        # true 클래스에 해당하는 확률 p_t
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        alpha_factor = tf.gather(alpha, y_true)  # shape: (batch_size,)
        loss_val = -alpha_factor * tf.pow(1 - p_t, gamma) * tf.math.log(p_t + 1e-7)
        return loss_val  # shape: (batch_size,)
    return loss

# =============================================================================
# 8. 학습 및 평가
# -----------------------------------------------------------------------------
def train_multimodal():
    for subject in SUBJECTS:
        print(f"\n=================== Training subject: {subject} ====================")
        eeg_data, eye_data, rppg_data, labels, seg_ids = load_all_modalities(subject)
        if rppg_data is None:
            continue
        
        # Stratified split (segment 단위)
        train_eeg, test_eeg, train_eye, test_eye, train_labels, test_labels = train_test_split(
            eeg_data, eye_data, labels, test_size=0.2, random_state=42, stratify=labels
        )
        rppg_train, rppg_test = train_test_split(rppg_data, test_size=0.2, random_state=42)
        
        # Train/Validation split
        train_eeg, valid_eeg, train_eye, valid_eye, train_labels, valid_labels = train_test_split(
            train_eeg, train_eye, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )
        rppg_train, rppg_valid = train_test_split(rppg_train, test_size=0.2, random_state=42)
        
        # 모든 모달의 train sample 수 동일하게 맞추기
        n_train = min(train_eeg.shape[0], train_eye.shape[0], rppg_train.shape[0])
        train_eeg = train_eeg[:n_train]
        train_eye = train_eye[:n_train]
        train_rppg = rppg_train[:n_train]
        train_labels = train_labels[:n_train]
        
        n_valid = min(valid_eeg.shape[0], valid_eye.shape[0], rppg_valid.shape[0])
        valid_eeg = valid_eeg[:n_valid]
        valid_eye = valid_eye[:n_valid]
        valid_rppg = rppg_valid[:n_valid]
        valid_labels = valid_labels[:n_valid]
        
        n_test = min(test_eeg.shape[0], test_eye.shape[0], rppg_test.shape[0])
        test_eeg = test_eeg[:n_test]
        test_eye = test_eye[:n_test]
        test_rppg = rppg_test[:n_test]
        test_labels = test_labels[:n_test]
        
        # 오버샘플링 적용
        train_eeg, train_eye, train_rppg, train_labels = oversample_data(train_eeg, train_eye, train_rppg, train_labels)
        print(f"After oversampling - Train Samples: {train_eeg.shape[0]}")
        
        alpha_values = compute_class_weights(train_labels)
        
        model = build_full_multimodal_model(num_classes=3)
        
        losses = {
            "Inter_EEG_RPPG": focal_loss(alpha=alpha_values),
            "Inter_EEG_Eye": focal_loss(alpha=alpha_values),
            "Inter_Eye_RPPG": focal_loss(alpha=alpha_values),
            "Intra_EEG": focal_loss(alpha=alpha_values),
            "Intra_Eye": focal_loss(alpha=alpha_values),
            "Intra_RPPG": focal_loss(alpha=alpha_values),
            # Final_Weights는 손실에 기여하지 않도록, 출력과 동일한 배치 크기의 0 텐서를 반환합니다.
            "Final_Weights": lambda y_true, y_pred: tf.zeros_like(tf.reduce_sum(y_pred, axis=-1))
        }
        loss_weights = {
            "Inter_EEG_RPPG": 1.0,
            "Inter_EEG_Eye": 1.0,
            "Inter_Eye_RPPG": 1.0,
            "Intra_EEG": 1.0,
            "Intra_Eye": 1.0,
            "Intra_RPPG": 1.0,
            "Final_Weights": 0.0
        }
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=losses,
            loss_weights=loss_weights,
            metrics={"Inter_EEG_RPPG": "accuracy"}
        )
        
        model.summary()
        
        # 각 branch에 대해 동일한 레이블을 사용 (Final_Weights는 더미값)
        train_labels_dict = {
            "Inter_EEG_RPPG": train_labels,
            "Inter_EEG_Eye": train_labels,
            "Inter_Eye_RPPG": train_labels,
            "Intra_EEG": train_labels,
            "Intra_Eye": train_labels,
            "Intra_RPPG": train_labels,
            "Final_Weights": np.zeros((len(train_labels), 6))
        }
        valid_labels_dict = {
            "Inter_EEG_RPPG": valid_labels,
            "Inter_EEG_Eye": valid_labels,
            "Inter_Eye_RPPG": valid_labels,
            "Intra_EEG": valid_labels,
            "Intra_Eye": valid_labels,
            "Intra_RPPG": valid_labels,
            "Final_Weights": np.zeros((len(valid_labels), 6))
        }
        test_labels_dict = {
            "Inter_EEG_RPPG": test_labels,
            "Inter_EEG_Eye": test_labels,
            "Inter_Eye_RPPG": test_labels,
            "Intra_EEG": test_labels,
            "Intra_Eye": test_labels,
            "Intra_RPPG": test_labels,
            "Final_Weights": np.zeros((len(test_labels), 6))
        }
        
        history = model.fit(
            [train_eeg, train_eye, train_rppg], train_labels_dict,
            validation_data=([valid_eeg, valid_eye, valid_rppg], valid_labels_dict),
            epochs=EPOCHS, batch_size=BATCH_SIZE
        )
        
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        model.save_weights(weight_path)
        print(f"✅ 모델 가중치 저장됨: {weight_path}")
        
        predictions = model.predict([test_eeg, test_eye, test_rppg])
        # 예측은 출력 순서에 따라 리스트로 반환되며, 여기서는 "Inter_EEG_RPPG" branch의 출력을 사용
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
