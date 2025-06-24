# CNN 기반의 Dual-Stream Feature Extractor + Cross-Modal Transformer

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
    LSTM, Add, Dropout, Softmax, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

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
SAVE_PATH = "/home/bcml1/sigenv/_2주차_eye+eeg_CNN/result2"
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
    x = Conv2D(16, kernel_size=(3,3), activation="relu", padding="same")(eeg_input)
    x = DepthwiseConv2D(kernel_size=(3,3), activation="relu", padding="same")(x)
    x = AveragePooling2D(pool_size=(2,2))(x)
    x = Conv2D(32, kernel_size=(3,3), activation="relu", padding="same")(x)
    x = Flatten()(x)
    eeg_output = Dense(128, activation="relu")(x)

    # 📌 Eye Crop Stream (500, 8, 64, 3)
    eye_input = Input(shape=(500, 8, 64, 3), name="Eye_Input")
    eye_cnn = TimeDistributed(Conv2D(16, kernel_size=(1,16), activation="relu", padding="same"))(eye_input)
    eye_cnn = TimeDistributed(DepthwiseConv2D(kernel_size=(4,6), activation="relu", padding="same"))(eye_cnn)
    eye_cnn = TimeDistributed(AveragePooling2D(pool_size=(3,2)))(eye_cnn)
    eye_cnn = TimeDistributed(Conv2D(32, kernel_size=(1,1), activation="relu", padding="same"))(eye_cnn)
    eye_cnn = TimeDistributed(Flatten())(eye_cnn)
    eye_cnn = TimeDistributed(Dense(64, activation="relu"))(eye_cnn)
    eye_lstm = LSTM(128, return_sequences=False, dropout=0.3)(eye_cnn)
    eye_output = Dense(128, activation="relu")(eye_lstm)

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
    
    # # ✅ 차원 확장 (Lambda 사용) -> 차원확장 제거
    # eeg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_query)  # (batch_size, 1, 128)
    # eye_key_value = Lambda(lambda x: tf.expand_dims(x, axis=1))(eye_key_value)  # (batch_size, 1, 128)

    # # 최종 Shape 확인
    # print("eeg_query shape after expand_dims:", eeg_query.shape)
    # print("eye_key_value shape after expand_dims:", eye_key_value.shape)
    eeg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(Dense(d_model)(eeg_features))  # (batch, 1, 128)
    eye_key_value = Lambda(lambda x: tf.expand_dims(x, axis=1))(Dense(d_model)(eye_features))


    cross_modal_attention_1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_EEG_to_Eye")(
        query=eeg_query, key=eye_key_value, value=eye_key_value
    )
    cross_modal_attention_1 = Dropout(dropout_rate)(cross_modal_attention_1)
    cross_modal_attention_1 = Add()([eeg_query, cross_modal_attention_1])
    cross_modal_attention_1 = LayerNormalization(epsilon=1e-6)(cross_modal_attention_1)

    # Cross-Modal Transformer: Eye Crop → EEG
    eye_query = Dense(d_model)(eye_features)
    eeg_key_value = Dense(d_model)(eeg_features)
    cross_modal_attention_2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_Eye_to_EEG")(
        query=eye_query, key=eeg_key_value, value=eeg_key_value
    )
    cross_modal_attention_2 = Dropout(dropout_rate)(cross_modal_attention_2)
    cross_modal_attention_2 = Add()([eye_query, cross_modal_attention_2])
    cross_modal_attention_2 = LayerNormalization(epsilon=1e-6)(cross_modal_attention_2)

    # # 차원 확장 (cross_modal_attention_2를 (None, 1, 128)로 변환) -> 제거
    # cross_modal_attention_2 = Lambda(lambda x: tf.expand_dims(x, axis=1))(cross_modal_attention_2)
    #cross_modal_attention_1 = tf.squeeze(cross_modal_attention_1, axis=1)
    cross_modal_attention_1 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_modal_attention_1)
    fused_features = Concatenate(axis=-1)([cross_modal_attention_1, cross_modal_attention_2])
    fused_features = Dense(d_model, activation="relu", name="Fused_Linear")(fused_features)

    # Self-Attention Transformer
    self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttentionFusion")(
        query=fused_features, key=fused_features, value=fused_features
    )
    self_attention = Dropout(dropout_rate)(self_attention)
    self_attention = Add()([fused_features, self_attention])
    self_attention = LayerNormalization(epsilon=1e-6)(self_attention)

    return self_attention

# -------------------------
# Intra-Modality Encoding Module
def create_intra_modality_encoding(eeg_features, eye_features, num_heads=4, d_model=128, dropout_rate=0.1):
    """
    EEG와 Eye Crop의 고유한 특성을 유지하며 강화하는 Intra-Modality Encoding Module
    """
    eeg_self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_EEG")(
        query=eeg_features, key=eeg_features, value=eeg_features
    )
    eeg_self_attention = Dropout(dropout_rate)(eeg_self_attention)
    eeg_self_attention = Add()([eeg_features, eeg_self_attention])
    eeg_self_attention = LayerNormalization(epsilon=1e-6)(eeg_self_attention)

    eye_self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_Eye")(
        query=eye_features, key=eye_features, value=eye_features
    )
    eye_self_attention = Dropout(dropout_rate)(eye_self_attention)
    eye_self_attention = Add()([eye_features, eye_self_attention])
    eye_self_attention = LayerNormalization(epsilon=1e-6)(eye_self_attention)

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
    eye_input = Input(shape=(500, 8, 64, 3), name="Eye_Input")
    
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
    
    print(f"fused feature shape:{fused_features}")
    # 예: fused_features, eeg_encoded, eye_encoded의 출력이 (batch, 1, 128)라면
    #fused_features = Lambda(lambda x: tf.squeeze(x, axis=1))(fused_features)
    #eeg_encoded = Lambda(lambda x: tf.squeeze(x, axis=1))(eeg_encoded)
    #eye_encoded = Lambda(lambda x: tf.squeeze(x, axis=1))(eye_encoded)          # (batch, 128)
    # 6️⃣ 분류 결과 결합을 위한 가중치(weight) 예측
    concat_features = Concatenate()([fused_features, eeg_encoded, eye_encoded])
    print("concat_features.shape:", concat_features.shape) #확인
    weights_logits  = Dense(3, name="Weight_Logits")(concat_features)  
    # weights         = Softmax(axis=-1, name="Weight_Softmax")(weights_logits)  # (batch_size, 3)
    # Softmax 레이어 대신 Lambda를 사용하여 softmax 적용
    weights = Lambda(lambda x: tf.nn.softmax(x, axis=-1), name="Weight_Softmax")(weights_logits)
    print("Weight_Logits.shape:", weights_logits.shape) #확인

    # ✅ 최종 모델 생성 (입력: 원본 Input / 출력: 세 개의 분류 결과와 가중치)
    model = Model(inputs=[eeg_input, eye_input],
                  outputs=[inter_classification, eeg_classification, eye_classification, weights],
                  name="Multimodal_Emotion_Classifier")
    
    # 🎯 **커스텀 멀티모달 손실 함수 정의**
    def multimodal_loss(y_true, y_preds):
        """
        L_total = W1 * L_inter + W2 * L_EEG + W3 * L_EyeCrop
        """
        inter_pred, eeg_pred, eye_pred, weights = y_preds  # 🎯 수정: weights_logits 대신 weights 직접 사용
        
        # 🎯 `y_true`가 (batch_size, ) 인 경우 변환 필요
        if len(y_true.shape) == 2 and y_true.shape[1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)  # (batch_size, ) 형태로 변환
            
        # 🎯 가중치 활용하여 손실 계산
        w1, w2, w3 = tf.split(weights, num_or_size_splits=3, axis=-1)  # 각각 분할
        loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
        loss_eeg   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eeg_pred)
        loss_eye   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eye_pred)

        # 🎯 최종 가중치 적용된 손실 계산
        total_loss = (w1 * loss_inter) + (w2 * loss_eeg) + (w3 * loss_eye)
        #return total_loss  # shape=(8,)
        return tf.reduce_mean(total_loss)   # shape=()
    
    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=multimodal_loss,
                  metrics=["accuracy"])
    
    return model

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
# 데이터 로드 함수
def load_multimodal_data(subject):
    eeg_data, eye_data, labels = [], [], []
    eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+).npy")
    
    for sample_index in range(1, 2):  # 예제에서는 Sample 01만 사용
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
            print(f"  🔍 Available frame indices: {frame_indices[:10]} ... {frame_indices[-10:]}")

            window_length = 500
            stride = 50
            candidate_windows = []
            for i in range(0, len(frame_indices) - window_length + 1, stride):
                window = frame_indices[i:i+window_length]
                candidate_windows.append(window)
            if len(candidate_windows) == 0:
                print(f"⚠ Warning: Not enough frames for sliding window for segment {segment_index:03d}. Skipping Eye Crop.")
                eye_data.append(None)
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
                continue

            selected_frames = min(candidate_windows, key=lambda w: abs(w[0] - expected_start))
            if len(selected_frames) < 500:
                print(f"⚠ Warning: Found only {len(selected_frames)} frames in selected window for segment {segment_index:03d}")
                while len(selected_frames) < 500:
                    selected_frames.append(selected_frames[-1])
                    print("프레임 복제됨")

            eye_frame_files = []
            for frame in selected_frames:
                if frame in file_mapping:
                    eye_frame_files.append(file_mapping[frame])
                if len(eye_frame_files) == 500:
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

            if len(eye_frame_stack) == 500:
                eye_data.append(np.stack(eye_frame_stack, axis=0))
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
            else:
                print(f"⚠ Warning: Found only {len(eye_frame_stack)} matching frames for segment {segment_index:03d}")

    print(f"✅ Total EEG Samples Loaded: {len(eeg_data)}")
    print(f"✅ Total Eye Crop Samples Loaded: {len([e for e in eye_data if e is not None])}")
    print(f"✅ Labels Loaded: {len(labels)}")
    return np.array(eeg_data), np.array([e if e is not None else np.zeros((500, 8, 64, 3)) for e in eye_data]), np.array(labels)

# -------------------------
# 학습 및 평가
def train_multimodal():
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    for subject in subjects:
        print(f"Training subject: {subject}")
        eeg_data, eye_data, labels = load_multimodal_data(subject)
        unique_samples = np.arange(len(eeg_data))
        train_samples, test_samples = train_test_split(unique_samples, test_size=0.2, random_state=42)
        train_samples, valid_samples = train_test_split(train_samples, test_size=0.2, random_state=42)
        train_eeg, train_eye, train_labels = eeg_data[train_samples], eye_data[train_samples], labels[train_samples]
        valid_eeg, valid_eye, valid_labels = eeg_data[valid_samples], eye_data[valid_samples], labels[valid_samples]
        test_eeg, test_eye, test_labels = eeg_data[test_samples], eye_data[test_samples], labels[test_samples]

        model = build_combined_model(num_classes=3)
        print(model.summary())

        train_labels = np.expand_dims(train_labels, axis=-1)
        valid_labels = np.expand_dims(valid_labels, axis=-1)
        print("train_labels의 shape: ", train_labels.shape)
        print("valid_labels의 shape: ", valid_labels.shape)
        
        start_epoch = 0
        max_epochs = 50
        batch_size = 2
        max_retries = 3

        for epoch in range(start_epoch, max_epochs):
            retries = 0
            while retries < max_retries:
                try:
                    print(f"\n🚀 Training {subject} - Epoch {epoch+1}/{max_epochs} (Retry: {retries})...")
                    model.fit(
                        [train_eeg, train_eye], train_labels,
                        validation_data=([valid_eeg, valid_eye], valid_labels),
                        epochs=1, batch_size=batch_size
                    )
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

        # 예측 결과 처리 부분
        predictions = model.predict([test_eeg, test_eye])
        inter_pred = predictions[0] # 평가에 사용할 분류 브랜치 선택 -> 오류 발생하지 않게 코드를 수정
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
