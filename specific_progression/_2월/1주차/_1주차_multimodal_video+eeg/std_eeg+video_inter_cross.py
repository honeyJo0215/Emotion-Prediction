# inter subject multi-modal transformer(뇌파+eye crop video)(cross 모달리티 사용 후 concat)
# (Cross-Modality 사용 후 Concat)

# 모달리티 적용 방식: EEG ↔ Eye Crop Video 간 Cross-Attention 학습 후 Concatenation
# Inter-Modality 처리 방식: EEG와 Eye Feature 간 Query-Key-Value Attention을 적용하여 상호작용 후 합침
# Concat 시점: Inter-Modality Attention (Cross-Attention) 후 Concatenation
# 특징 강조 방식: EEG ↔ Eye Video 간 상관성을 학습하여 중요한 정보 교환 후 특징 강화

# EEG와 Eye Crop Video 데이터에서 각각 특징을 추출한 후,
# EEG의 Query와 Eye의 Key/Value, 그리고 Eye의 Query와 EEG의 Key/Value를 이용한 Cross-Attention을 통해 두 모달리티 간의 상관관계를 직접적으로 학습합니다.
# 그 후, Cross-Attention 결과를 Concatenation하여 Fusion하고, 최종 분류에 활용합니다.

import os
import re
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Conv3D, BatchNormalization, Dropout, LayerNormalization, Lambda, GlobalAveragePooling1D, Reshape
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# ---------------------------------
# 1. 데이터 경로 및 감정 라벨 매핑
# ---------------------------------

EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
SAVE_PATH = "/home/bcml1/sigenv/_multimodal_video+eeg/result_stand_modelinter_cross"
os.makedirs(SAVE_PATH, exist_ok=True)

# 감정 라벨 매핑
EMOTION_MAPPING = {
    "Excited": 0,
    "Relaxed": 1,
    "Stressed": 2,
    "Bored": 3,
    "Neutral": 4
}

# =============================================================================
# 1. EEG Branch 관련 모듈
# =============================================================================
# 1-1. Spatial-Spectral Convolution Module
class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.spatial_conv = Conv3D(filters, kernel_size=(1, 3, 3), strides=strides, padding="same", activation="relu")
        self.spectral_conv = Conv3D(filters, kernel_size=(4, 1, 1), strides=strides, padding="same", activation="relu")

    def call(self, inputs):
        if len(inputs.shape) == 4:
            inputs = tf.expand_dims(inputs, axis=-1)
        spatial_features = self.spatial_conv(inputs)
        spectral_features = self.spectral_conv(inputs)
        return spatial_features + spectral_features
    
    def get_config(self):
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
        }

# 1-2. Spatial and Spectral Attention Branch
class SpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialSpectralAttention, self).__init__()
        self.spatial_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")
        self.spectral_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")

    def call(self, inputs):
        if len(inputs.shape) == 4:
            inputs = tf.expand_dims(inputs, axis=-1)

        # Spatial attention
        spatial_mask = self.spatial_squeeze(inputs)
        spatial_output = inputs * spatial_mask

        # Spectral attention
        spectral_mask = self.spectral_squeeze(inputs)
        spectral_output = inputs * spectral_mask

        # Combine spatial and spectral outputs
        combined_output = spatial_output + spectral_output
        return combined_output

    def get_config(self):
        return {}

# 1-3. EEG CNN 모델 (EEG의 공간-주파수-시간적 특징 추출)
class EEGCNN(tf.keras.Model):
    def __init__(self, d_model):
        super(EEGCNN, self).__init__()
        self.attention = SpatialSpectralAttention()
        self.conv1 = SpatialSpectralConvModule(8, kernel_size=(1,3,3), strides=(1,3,3))
        self.conv2 = SpatialSpectralConvModule(16, kernel_size=(4,1,1), strides=(4,1,1))
        self.conv3 = SpatialSpectralConvModule(32, kernel_size=(1,2,2), strides=(1,2,2))
        self.flatten = Flatten()
        self.dense_proj = Dense(d_model, activation="relu")
    def call(self, inputs):
        x = self.attention(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense_proj(x)
        return x

# =============================================================================
# 2. Eye Crop Video Branch 관련 모듈
# =============================================================================
def build_eye_crop_model(d_model=64):
    """
    Eye Crop Video 데이터 (예: 50프레임, 8x64, 3채널)를 입력으로 받아
    Conv3D와 BatchNormalization을 통해 지역 특징을 추출하고, Dense layer로 d_model 차원으로 매핑합니다.
    """
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
# 3. Transformer Encoder 및 Cross-Attention 모듈
# =============================================================================
# 3-1. Transformer Encoder Layer (Self-Attention 기반)
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation=tf.nn.gelu),
            Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        if len(inputs.shape) == 2:  # (batch, features)
            inputs = tf.expand_dims(inputs, axis=1)  # (batch, 1, features)
        
        query, key, value = inputs, inputs, inputs  # 같은 입력을 사용하여 self-attention 수행
        attn_output = self.mha(query=query, key=key, value=value, training=training)
        
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual Connection

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # 최종 Residual Connection

# 3-2. Transformer Encoder (여러 레이어를 쌓음)
def build_transformer_encoder(seq_length, n_layers=2, n_heads=4, d_ff=512, dropout_rate=0.1, d_model=64, name="Transformer_Encoder"):
    inputs = Input(shape=(seq_length, d_model))
    x = inputs
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate)(x)
    return Model(inputs, x, name=name)

# ----------------------------------------------------
# 3. Inter → Intra 학습 구조 구현 (Cross)
# ----------------------------------------------------
def build_multimodal_model(eeg_input_shape, eye_input_shape=(50,8,64,3), seq_length=16, d_model=64):
    # 입력 정의
    eeg_input = Input(shape=eeg_input_shape, name="EEG_Input")
    eye_input = Input(shape=eye_input_shape, name="Eye_Input")
    
    # 모달리티별 Feature Extraction (CNN)
    f_eeg = EEGCNN(d_model=d_model)(eeg_input)           # (batch, d_model)
    f_eye = build_eye_crop_model(d_model=d_model)(eye_input)  # (batch, d_model)
    
    # 시퀀스로 확장 (Tiling)
    tile_layer = Lambda(lambda x: tf.tile(tf.expand_dims(x, axis=1), [1, seq_length, 1]))
    eeg_seq = tile_layer(f_eeg)  # (batch, seq_length, d_model)
    eye_seq = tile_layer(f_eye)  # (batch, seq_length, d_model)
    
    # 1. Inter-Modality Attention (Cross-Attention 적용)
    # EEG의 Query, Eye의 Key/Value
    cross_eeg = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model)(query=eeg_seq, key=eye_seq, value=eye_seq)
    # Eye의 Query, EEG의 Key/Value
    cross_eye = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model)(query=eye_seq, key=eeg_seq, value=eeg_seq)
    # Residual 연결과 Layer Normalization 적용
    eeg_cross = LayerNormalization(epsilon=1e-6)(eeg_seq + cross_eeg)
    eye_cross = LayerNormalization(epsilon=1e-6)(eye_seq + cross_eye)
    
    # 2. Intra-Modality Attention (각 모달리티 내부 Self-Attention 적용)
    transformer_eeg_intra = build_transformer_encoder(seq_length, n_layers=2, n_heads=4, d_ff=512, dropout_rate=0.1, d_model=d_model, name="Transformer_EEG_Intra")
    transformer_eye_intra = build_transformer_encoder(seq_length, n_layers=2, n_heads=4, d_ff=512, dropout_rate=0.1, d_model=d_model, name="Transformer_EYE_Intra")
    eeg_intra = transformer_eeg_intra(eeg_cross)   # (batch, seq_length, d_model)
    eye_intra = transformer_eye_intra(eye_cross)   # (batch, seq_length, d_model)
    
    # 각 모달리티의 최종 표현은 Global Average Pooling으로 집계
    eeg_final = GlobalAveragePooling1D()(eeg_intra)  # (batch, d_model)
    eye_final = GlobalAveragePooling1D()(eye_intra)  # (batch, d_model)
    
    # 3. Concatenation & Classification
    fused = Concatenate(axis=-1)([eeg_final, eye_final])  # (batch, 2*d_model)
    x = Dense(128, activation="relu")(fused)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(5, activation="softmax")(x)  # 5 감정 클래스
    
    # # 🚀 Mean Pooling (Transformer Sequence Length 축소)
    # output = Lambda(lambda x: tf.reduce_mean(x, axis=1))(output)  # (batch_size, num_classes)

    model = Model(inputs=[eeg_input, eye_input], outputs=output, name="Multimodal_CrossTransformer_Model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    
    return model

# ---------------------------------
# 4. Intra-subject Cross-Validation 코드 (Sample 기준)
# ---------------------------------

def find_subject_folder(base_path, subject):
    """실제 파일 시스템에서 subject(s01, s02 ...)에 해당하는 폴더를 찾음."""
    possible_folders = os.listdir(base_path)  # eye_crop 내 폴더 확인
    for folder in possible_folders:
        if folder.lower() == subject.lower():  # 대소문자 무시하고 비교
            return os.path.join(base_path, folder)
    return None  # 해당 폴더를 찾지 못한 경우

# ✅ **입력 데이터 크기 줄이기 (다운샘플링)**
def downsample_eye_frame(frame):
    """Eye Crop 이미지 다운샘플링 (64x32 → 32x16)"""
    return cv2.resize(frame, (32,8), interpolation=cv2.INTER_AREA)  # 해상도 절반 감소

# ✅ **Eye Crop 데이터 로드 시 다운샘플링 적용**
def reshape_eye_frame(data):
    """
    (N, 32, 64, 3) 형태의 eye frame 데이터를 (32, 64, 3)으로 변환 후 다운샘플링 적용.
    - N이 2 이상이면 평균을 내서 병합.
    - N이 1이면 그대로 사용.
    """
    if len(data.shape) == 4 and data.shape[0] > 0:  
        reshaped_data = np.mean(data, axis=0)  # 모든 요소를 평균 내어 병합 (32, 64, 3)
        return downsample_eye_frame(reshaped_data)  # 다운샘플링 적용
    elif len(data.shape) == 3 and data.shape == (32, 64, 3):  
        return downsample_eye_frame(data)  # 다운샘플링 적용
    else:
        raise ValueError(f"Unexpected shape: {data.shape}, expected (N, 32, 64, 3) or (32, 64, 3)")

def load_multimodal_data(subject):
    eeg_data, eye_data, labels = [], [], []

    eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+)_2D_DE.npy")

    for sample_index in range(40):  # Sample 000 ~ Sample 040까지 40->3로 수정함
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
                        print(f"⚠ Skipping invalid file name: {f} (No trial/frame pattern found)")
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
            print(f"  🔍 Available frame indices: {frame_indices[:10]} ... {frame_indices[-10:]}")  

            selected_frames = sorted([frame for frame in frame_indices if start_frame <= frame < end_frame])

            if len(selected_frames) == 0:
                print(f"⚠ Warning: No frames found for segment {segment_index}. Skipping Eye Crop.")
                eye_data.append(None)  # Eye Crop 데이터 없음
                eeg_data.append(eeg_segment)  # EEG 데이터는 추가
                labels.append(EMOTION_MAPPING[emotion_label])  
                continue

            if len(selected_frames) < 50:
                print(f"⚠ Warning: Found only {len(selected_frames)} frames for segment {segment_index}")
                while len(selected_frames) < 50:
                    selected_frames.append(selected_frames[-1])  # 부족한 프레임을 마지막 프레임으로 채움
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
                # 📌 만약 32x64로 로드되었다면 64x64로 맞추기 위해 padding 적용
                if frame_data.shape[-2] == 32:  # 🚨 너비가 32인 경우
                    pad_width = [(0, 0)] * frame_data.ndim  # 기존 shape 유지
                    pad_width[-2] = (16, 16)  # 🚀 너비(32→64) 확장
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

    return np.array(eeg_data), np.array([e if e is not None else np.zeros((50, 8, 64, 3)) for e in eye_data]), np.array(labels)

# 🟢 **학습 및 평가**
def train_multimodal():
    subjects = [f"s{str(i).zfill(2)}" for i in range(13, 23)]
    for subject in subjects:
        print(f"Training subject: {subject}")
        
        # 데이터 로드
        eeg_data, eye_data, labels = load_multimodal_data(subject)

        # 샘플 단위로 Train/Valid/Test 데이터 나누기
        unique_samples = np.arange(len(eeg_data))  
        train_samples, test_samples = train_test_split(unique_samples, test_size=0.2, random_state=42)
        train_samples, valid_samples = train_test_split(train_samples, test_size=0.2, random_state=42)

        # 샘플 인덱스 기반으로 데이터 분할
        train_eeg, train_eye, train_labels = eeg_data[train_samples], eye_data[train_samples], labels[train_samples]
        valid_eeg, valid_eye, valid_labels = eeg_data[valid_samples], eye_data[valid_samples], labels[valid_samples]
        test_eeg, test_eye, test_labels = eeg_data[test_samples], eye_data[test_samples], labels[test_samples]

        # 모델이 학습 도중 OOM으로 종료될 경우 체크포인트를 저장하고 재시작하면 메모리 문제를 해결가능
        # 🚀 **각 subject 별 체크포인트 저장 경로 설정**
        checkpoint_dir = f"/home/bcml1/sigenv/_multimodal_video+eeg/checkpoint/cross_{subject}"
        checkpoint_path = os.path.join(checkpoint_dir, "cp.weights.h5")
        os.makedirs(checkpoint_dir, exist_ok=True)  # 디렉토리 없으면 생성

        # 체크포인트 콜백 설정 (자동 저장)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        )
        
        # 멀티모달 모델 구축
        model = build_multimodal_model(eeg_input_shape=train_eeg.shape[1:])
        print(model.summary())

        # 🚀 **기존 체크포인트 로드 (있다면)**
        if os.path.exists(checkpoint_path + ".index"):
            print(f"✅ Checkpoint found for {subject}, loading model...")
            model.load_weights(checkpoint_path)

        # 라벨 차원 확장
        train_labels = np.expand_dims(train_labels, axis=-1)
        valid_labels = np.expand_dims(valid_labels, axis=-1)

        # 🚀 **학습 파라미터 설정**
        start_epoch = 0
        max_epochs = 50
        batch_size = 2
        max_retries = 3  # 한 에포크당 최대 재시도 횟수

        # 에포크별 학습
        for epoch in range(start_epoch, max_epochs):
            retries = 0
            while retries < max_retries:
                try:
                    print(f"\n🚀 Training {subject} - Epoch {epoch+1}/{max_epochs} (Retry: {retries})...")
                    model.fit(
                        [train_eeg, train_eye], train_labels,
                        validation_data=([valid_eeg, valid_eye], valid_labels),
                        epochs=1, batch_size=batch_size,
                        callbacks=[checkpoint_callback]
                    )
                    # 에포크가 정상적으로 완료되면 while 루프 탈출
                    break  
                except tf.errors.ResourceExhaustedError:
                    print(f"⚠️ OOM 발생! 체크포인트 저장 후 GPU 메모리 정리 & 재시작 (Retry: {retries+1})...")
                    # 체크포인트 저장 시 OOM이 발생할 경우 예외 처리
                    try:
                        model.save_weights(checkpoint_path)
                    except tf.errors.ResourceExhaustedError:
                        print("⚠️ 체크포인트 저장 중 OOM 발생 - 저장 건너뜀.")
                    
                    # GPU 메모리 정리
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    # 모델 재생성 및 체크포인트 로드 (있다면)
                    model = build_multimodal_model(eeg_input_shape=train_eeg.shape[1:])
                    if os.path.exists(checkpoint_path + ".index"):
                        model.load_weights(checkpoint_path)
                    
                    retries += 1
                    # 재시도 전에 잠시 휴식 (옵션)
                    tf.keras.backend.sleep(1)
            else:
                # 최대 재시도 횟수를 초과하면 에포크 종료 및 다음 subject로 넘어감.
                print(f"❌ 에포크 {epoch+1}에서 최대 재시도 횟수를 초과하였습니다. 다음 subject로 넘어갑니다.")
                break  # 또는 continue를 사용하여 다음 subject로 넘어갈 수 있음.

        # 🚀 **최종 모델 저장**
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        model.save_weights(weight_path)
        print(f"✅ 모델 가중치 저장됨: {weight_path}")

        # 🚀 **테스트 평가**
        predictions = model.predict([test_eeg, test_eye])
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
        
if __name__ == "__main__":
    train_multimodal()