#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eye 데이터만을 사용하여 학습하는 모델

- Eye 데이터를 CNN으로 피처 추출한 후 Transformer Encoder로 융합
- Positional Encoding을 통해 시퀀스 순서 정보를 반영
- CNN 모델에 BatchNormalization과 Dropout을 추가하여 일반화 성능 향상
- Test 데이터 분할은 sample 기준 그대로 유지

각 섹션은 필요에 따라 수정할 수 있습니다.
"""

import os
import re
import cv2
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Reshape, Input, Dense, Flatten, Conv3D, BatchNormalization, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =============================================================================
# GPU 메모리 제한 설정 (필요 시 수정)
# =============================================================================
# def limit_gpu_memory(memory_limit_mib=8000):
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             tf.config.experimental.set_virtual_device_configuration(
#                 gpus[0],
#                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
#             )
#             print(f"GPU memory limited to {memory_limit_mib} MiB.")
#         except RuntimeError as e:
#             print(e)
#     else:
#         print("No GPU available, using CPU.")

# limit_gpu_memory(8000)

# =============================================================================
# 데이터 경로 및 감정 라벨 매핑 (환경에 맞게 수정)
# =============================================================================
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"  # EEG 관련 경로 (사용하지 않음)
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
SAVE_PATH = "/home/bcml1/myenv/DEAP_EyeOnly"
os.makedirs(SAVE_PATH, exist_ok=True)

EMOTION_MAPPING = {
    "Excited": 0,
    "Relaxed": 1,
    "Stressed": 2,
    "Bored": 3,
    "Neutral": 4
}

# =============================================================================
# Positional Encoding
# =============================================================================
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(self.sequence_length, self.d_model),
            initializer="uniform",
            trainable=True
        )
        super(PositionalEncoding, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, sequence_length, d_model)
        return inputs + self.pos_embedding

# =============================================================================
# Eye Crop CNN 모델
# =============================================================================
def build_eye_crop_model(d_model=64):
    eye_input = Input(shape=(50, 8, 64, 3))  # 입력 크기 (프레임 수, 높이, 너비, 채널)
    x = Reshape((50, 8, 64, 3))(eye_input)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(d_model, activation='relu')(x)
    x = Dropout(0.3)(x)
    return Model(eye_input, x, name="EyeCrop_CNN")

# =============================================================================
# Transformer Encoder Layer
# =============================================================================
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
        # Multi-Head Self-Attention
        attn_output = self.mha(query=inputs, key=inputs, value=inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        # Feed Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# =============================================================================
# Transformer Encoder 모델 (여러 층 적용)
# =============================================================================
def build_transformer_encoder(input_dim, n_layers=2, n_heads=4, d_ff=512,
                              dropout_rate=0.1, d_model=64, name="Transformer_Encoder"):
    inputs = Input(shape=(input_dim, d_model))
    x = inputs
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model=d_model, n_heads=n_heads,
                                    d_ff=d_ff, dropout_rate=dropout_rate)(x)
    return Model(inputs, x, name=name)

# =============================================================================
# Eye Only 모델 구성
# =============================================================================
def build_eye_only_model(eye_input_shape=(50, 8, 64, 3)):
    # 입력 정의 (Eye 데이터만 사용)
    eye_input = Input(shape=eye_input_shape)
    
    # Eye Crop CNN 적용
    eye_cnn_model = build_eye_crop_model(d_model=64)
    eye_features = eye_cnn_model(eye_input)  # (batch, 64)
    
    # 시퀀스 형태로 확장 (길이 64, feature dimension 64)
    expand_dims_layer = Lambda(lambda x: tf.tile(x[:, None, :], [1, 64, 1]))
    eye_seq = expand_dims_layer(eye_features)  # (batch, 64, 64)
    
    # Positional Encoding 적용
    eye_seq = PositionalEncoding(sequence_length=64, d_model=64)(eye_seq)
    
    # Transformer Encoder 적용 (Intra-modality)
    eye_transformer = build_transformer_encoder(input_dim=64, name="Transformer_Encoder_EYE")
    eye_transformed = eye_transformer(eye_seq)  # (batch, 64, 64)
    
    # Classification Head
    x = Dense(128, activation="relu")(eye_transformed)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(5, activation="softmax")(x)
    # Mean Pooling으로 시퀀스 축소 → 최종 출력 (batch, 5)
    output = Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)
    
    model = Model(inputs=eye_input, outputs=output, name="Eye_Only_Model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    return model

# =============================================================================
# 데이터 전처리 및 로드 함수들 (필요에 따라 수정)
# =============================================================================
def downsample_eye_frame(frame):
    """Eye Crop 이미지 다운샘플링 (예: 64x32 → 32x8)"""
    return cv2.resize(frame, (32, 8), interpolation=cv2.INTER_AREA)

def reshape_eye_frame(data):
    """
    (N, 32, 64, 3) 형태의 eye frame 데이터를 처리.
    N이 2 이상이면 평균내고, 1이면 그대로 사용 후 다운샘플링.
    """
    if len(data.shape) == 4 and data.shape[0] > 0:
        reshaped_data = np.mean(data, axis=0)
        return downsample_eye_frame(reshaped_data)
    elif len(data.shape) == 3 and data.shape == (32, 64, 3):
        return downsample_eye_frame(data)
    else:
        raise ValueError(f"Unexpected shape: {data.shape}, expected (N, 32, 64, 3) or (32, 64, 3)")

def load_multimodal_data(subject):
    """
    지정한 subject의 EEG 및 Eye Crop 데이터를 로드.
    본 함수에서는 EEG 데이터는 로드하나, Eye Only 모델 학습을 위해 eye 데이터와 라벨만 사용합니다.
    파일명 및 경로 규칙은 데이터셋에 맞게 수정.
    """
    eeg_data, eye_data, labels = [], [], []
    eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+)_2D_DE.npy")

    for sample_index in range(40):  # Sample 000 ~ Sample 039
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

            # 초기 세그먼트 (3개 미만) 무시
            if segment_index < 3:
                continue
            segment_index -= 3

            eeg_file_path = os.path.join(EEG_DATA_PATH, file_name)
            eeg_segment = np.load(eeg_file_path)

            eye_subject_path = os.path.join(EYE_CROP_PATH, subject)
            if not os.path.exists(eye_subject_path):
                print(f"🚨 Subject folder not found: {eye_subject_path}")
                continue

            trial_number = sample_index + 1
            start_frame = segment_index * 50
            end_frame = start_frame + 50

            print(f"  🔹 Segment {segment_index}: Expected frames {start_frame} to {end_frame}, Matching Trial {trial_number:02d}")

            frame_indices = set()
            file_mapping = {}
            for f in os.listdir(eye_subject_path):
                try:
                    if not f.startswith(subject) or not f.endswith(".npy"):
                        continue
                    match = re.search(r"trial(\d+).avi_frame(\d+)", f)
                    if not match:
                        print(f"⚠ Skipping invalid file name: {f}")
                        continue
                    file_trial_number = int(match.group(1))
                    frame_number = int(match.group(2))
                    if file_trial_number == trial_number:
                        frame_indices.add(frame_number)
                        file_mapping[frame_number] = os.path.join(eye_subject_path, f)
                except ValueError as e:
                    print(f"🚨 Error processing file {f}: {e}")
                    continue

            frame_indices = sorted(frame_indices)
            if frame_indices:
                print(f"  🔍 Available frame indices: {frame_indices[:10]} ... {frame_indices[-10:]}")
            else:
                print("  ⚠ No frame indices found.")

            selected_frames = sorted([frame for frame in frame_indices if start_frame <= frame < end_frame])
            if len(selected_frames) == 0:
                print(f"⚠ Warning: No frames found for segment {segment_index}. Skipping Eye Crop.")
                eye_data.append(None)
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
                continue

            if len(selected_frames) < 50:
                print(f"⚠ Warning: Found only {len(selected_frames)} frames for segment {segment_index}")
                while len(selected_frames) < 50:
                    selected_frames.append(selected_frames[-1])
                    print("프레임 복제됨")

            eye_frame_files = []
            for frame in selected_frames:
                if frame in file_mapping:
                    eye_frame_files.append(file_mapping[frame])
                if len(eye_frame_files) == 50:
                    break

            eye_frame_stack = []
            for f in eye_frame_files:
                frame_data = np.load(f)
                frame_data = reshape_eye_frame(frame_data)
                # 만약 프레임 너비가 32이면 패딩으로 64로 확장
                if frame_data.shape[-2] == 32:
                    pad_width = [(0, 0)] * frame_data.ndim
                    pad_width[-2] = (16, 16)
                    frame_data = np.pad(frame_data, pad_width, mode='constant', constant_values=0)
                eye_frame_stack.append(frame_data)

            if len(eye_frame_stack) == 50:
                eye_data.append(np.stack(eye_frame_stack, axis=0))
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
            else:
                print(f"⚠ Warning: Found only {len(eye_frame_stack)} matching frames for segment {segment_index}")

    print(f"✅ Total EEG Samples Loaded: {len(eeg_data)}")
    print(f"✅ Total Eye Crop Samples Loaded: {len([e for e in eye_data if e is not None])}")
    print(f"✅ Labels Loaded: {len(labels)}")
    # None 값은 모두 zeros로 대체 (눈 데이터만 사용)
    eye_data = np.array([e if e is not None else np.zeros((50, 8, 64, 3)) for e in eye_data])
    return np.array(eeg_data), eye_data, np.array(labels)

# =============================================================================
# 학습 및 평가 함수 (Eye 데이터만 사용)
# =============================================================================
def train_eye_only():
    subjects = [f"s{str(i).zfill(2)}" for i in range(17, 23)]
    
    for subject in subjects:
        print(f"\n===== Training subject: {subject} =====")

        # 데이터 로드 (EEG 데이터는 무시)
        _, eye_data, labels = load_multimodal_data(subject)

        # 샘플 단위 Train/Validation/Test 분할
        unique_samples = np.arange(len(eye_data))
        train_samples, test_samples = train_test_split(unique_samples, test_size=0.2, random_state=42)
        train_samples, valid_samples = train_test_split(train_samples, test_size=0.2, random_state=42)

        train_eye, valid_eye, test_eye = eye_data[train_samples], eye_data[valid_samples], eye_data[test_samples]
        train_labels, valid_labels, test_labels = labels[train_samples], labels[valid_samples], labels[test_samples]

        # 각 subject 별 체크포인트 저장 경로 설정
        checkpoint_dir = f"/home/bcml1/myenv/DEAP_EyeOnly_checkpoint/{subject}"
        checkpoint_path = os.path.join(checkpoint_dir, "cp.weights.h5")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 체크포인트 콜백 설정 (자동 저장)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        # 모델 생성 (Eye Only)
        model = build_eye_only_model(eye_input_shape=train_eye.shape[1:])
        print(model.summary())

        # 기존 체크포인트 로드 (있다면)
        if os.path.exists(checkpoint_path + ".index"):
            print(f"✅ Checkpoint found for {subject}, loading model...")
            model.load_weights(checkpoint_path)

        # 라벨 차원 확장 (필요한 경우)
        train_labels = np.expand_dims(train_labels, axis=-1)
        valid_labels = np.expand_dims(valid_labels, axis=-1)

        # 학습 파라미터 설정
        start_epoch = 0
        max_epochs = 50
        batch_size = 2
        max_retries = 5  # 한 에포크당 최대 재시도 횟수

        # 에포크별 학습
        for epoch in range(start_epoch, max_epochs):
            retries = 0
            while retries < max_retries:
                try:
                    print(f"\n🚀 Training {subject} - Epoch {epoch+1}/{max_epochs} (Retry: {retries})...")
                    model.fit(
                        train_eye, train_labels,
                        validation_data=(valid_eye, valid_labels),
                        epochs=1, batch_size=batch_size,
                        #callbacks=[checkpoint_callback]
                    )
                    # 에포크가 정상적으로 완료되면 while 루프 탈출
                    break  
                except tf.errors.ResourceExhaustedError:
                    print(f"⚠️ OOM 발생! 체크포인트 저장 후 GPU 메모리 정리 & 재시작 (Retry: {retries+1})...")
                    try:
                        model.save_weights(checkpoint_path)
                    except tf.errors.ResourceExhaustedError:
                        print("⚠️ 체크포인트 저장 중 OOM 발생 - 저장 건너뜀.")
                    
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    model = build_eye_only_model(eye_input_shape=train_eye.shape[1:])
                    if os.path.exists(checkpoint_path + ".index"):
                        model.load_weights(checkpoint_path)
                    
                    retries += 1
                    tf.keras.backend.sleep(1)
            else:
                print(f"❌ 에포크 {epoch+1}에서 최대 재시도 횟수를 초과하였습니다. 다음 subject로 넘어갑니다.")
                break

        # 최종 모델 저장
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_eye_only_model.weights.h5")
        model.save_weights(weight_path)
        print(f"✅ 모델 가중치 저장됨: {weight_path}")

        # 테스트 평가
        predictions = model.predict(test_eye)
        predicted_labels = np.argmax(predictions, axis=-1)
        test_report = classification_report(
            test_labels, predicted_labels,
            target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"],
            labels=[0, 1, 2, 3, 4],
            zero_division=0
        )
        print(f"\n📊 Test Report for {subject}")
        print(test_report)

        report_path = os.path.join(subject_save_path, f"{subject}_test_report.txt")
        with open(report_path, "w") as f:
            f.write(test_report)
        print(f"✅ 테스트 리포트 저장됨: {report_path}")

# =============================================================================
# 메인 실행부
# =============================================================================
if __name__ == "__main__":
    train_eye_only()
