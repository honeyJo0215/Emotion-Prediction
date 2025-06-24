#inter subject multi-modal transformer(뇌파+eye crop video):단순한 inter 모델

import os
import re
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Conv3D, BatchNormalization, Dropout, LayerNormalization, Lambda, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =============================================================================
# 0. GPU 메모리 제한 (필요 시)
# =============================================================================
def limit_gpu_memory(memory_limit_mib=1500):
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

limit_gpu_memory(1500)

# ---------------------------------
# 1. 데이터 경로 및 감정 라벨 매핑
# ---------------------------------

EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
SAVE_PATH = "/home/bcml1/sigenv/_multimodal_video+eeg/result_stand_inter_concat"
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
# 1. 유틸리티 함수
# =============================================================================
def preprocess_data(inputs):
    """
    입력 데이터가 4D (batch, D, H, W)인 경우 채널 차원 추가.
    EEG 데이터가 이미 2D mapping된 경우 (4,6,6) 형태라면 채널 차원을 추가하여 (4,6,6,1)로 만듭니다.
    """
    if len(inputs.shape) == 4:
        inputs = tf.expand_dims(inputs, axis=-1)
    return inputs

# =============================================================================
# 2. 제공된 모듈들
# =============================================================================
# 2-1. Spatial-Spectral Convolution Module
class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.spatial_conv = Conv3D(filters,
                                   kernel_size=(1, 3, 3),
                                   strides=strides,
                                   padding="same",
                                   activation="relu")
        self.spectral_conv = Conv3D(filters,
                                    kernel_size=(4, 1, 1),
                                    strides=strides,
                                    padding="same",
                                    activation="relu")

    def call(self, inputs):
        # 입력이 4D라면 (batch, D, H, W) -> (batch, D, H, W, 1)
        if len(inputs.shape) == 4:
            inputs = tf.expand_dims(inputs, axis=-1)
        spatial_features = self.spatial_conv(inputs)
        spectral_features = self.spectral_conv(inputs)
        return spatial_features + spectral_features

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
        })
        return config

# 2-2. Spatial and Spectral Attention Branch
class SpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialSpectralAttention, self).__init__()
        self.spatial_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")
        self.spectral_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")

    def call(self, inputs):
        if len(inputs.shape) == 4:
            inputs = tf.expand_dims(inputs, axis=-1)
    
        spatial_mask = self.spatial_squeeze(inputs)
        spatial_output = inputs * spatial_mask

        spectral_mask = self.spectral_squeeze(inputs)
        spectral_output = inputs * spectral_mask

        combined_output = spatial_output + spectral_output
        return combined_output

    def get_config(self):
        return {}

# 2-3. Transformer Encoder Layer
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
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
        # 만약 inputs가 (batch, d_model) shape이면 (batch, 1, d_model)로 확장합니다.
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
        })
        return config

# =============================================================================
# 3. 모달리티별 Feature Extractor 및 Fusion 모듈
# =============================================================================
# 3-1. EEG Feature Extractor (Transformer Encoder 기반)
class EEGTransformerEncoder(Model):
    def __init__(self, d_model=64, n_layers=6, n_heads=8, d_ff=2048, dropout_rate=0.5):
        """
        EEG 입력 (DE feature mapping된 2D 데이터)을 받아, Spatial-Spectral Attention와 3개의 Conv3D 블록, 
        Flatten, Dense Projection 후 다층 Transformer Encoder Layer를 통과시켜 feature vector를 추출합니다.
        """
        super(EEGTransformerEncoder, self).__init__()
        self.attention = SpatialSpectralAttention()
        self.conv_block1 = SpatialSpectralConvModule(8, kernel_size=(1, 3, 3), strides=(1, 3, 3))
        self.conv_block2 = SpatialSpectralConvModule(16, kernel_size=(4, 1, 1), strides=(4, 1, 1))
        self.conv_block3 = SpatialSpectralConvModule(32, kernel_size=(1, 2, 2), strides=(1, 2, 2))
        self.flatten = Flatten()
        self.dense_projection = Dense(d_model, activation="relu")
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=dropout_rate)
            for _ in range(n_layers)
        ]
    
    def call(self, inputs, training=False):
        x = preprocess_data(inputs)  # (batch, 4,6,6,1)
        x = self.attention(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense_projection(x)  # shape: (batch, d_model)
        # 각 Transformer layer는 sequence 입력을 요구하므로 (batch, 1, d_model)로 확장 후 처리하고 다시 squeeze 합니다.
        for layer in self.encoder_layers:
            x = layer(tf.expand_dims(x, axis=1), training=training)
            x = tf.squeeze(x, axis=1)
        return x  # EEG feature vector (batch, d_model)

# 3-2. Eye Feature Extractor (ResNet50 기반)
class EyeFeatureExtractor(Model):
    def __init__(self):
        super(EyeFeatureExtractor, self).__init__()
        base_model = tf.keras.applications.ResNet50(include_top=False,
                                                    weights="imagenet",
                                                    input_shape=(64, 32, 3))
        self.feature_extractor = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D()
        ])

    def call(self, x, training=False):
        return self.feature_extractor(x, training=training)

# 3-3. Fusion Module: EEG와 Eye의 feature 및 Subject embedding을 결합 (Transformer Encoder Layer 활용)
class FusionModule(Model):
    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout_rate=0.1, num_subjects=10):
        """
        inter-subject setting을 위해 subject id를 Embedding한 토큰을 추가합니다.
        num_subjects: 전체 피험자 수
        """
        super(FusionModule, self).__init__()
        # Eye feature의 차원을 EEG feature와 맞추기 위한 projection layer
        self.eye_projection = Dense(d_model, activation="relu")
        self.transformer_layer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate)
        self.classifier = Dense(5, activation="softmax")
        # subject id를 임베딩하여 d_model 차원의 토큰으로 변환
        self.subject_embedding = tf.keras.layers.Embedding(input_dim=num_subjects, output_dim=d_model)
    
    def call(self, eeg_features, eye_features, subject_ids, training=False):
        eye_proj = self.eye_projection(eye_features)  # (batch, d_model)
        subject_token = self.subject_embedding(subject_ids)  # (batch, d_model)
        # 두 모달리티와 subject 토큰을 sequence의 세 토큰으로 취급합니다.
        fusion_seq = tf.stack([eeg_features, eye_proj, subject_token], axis=1)  # shape: (batch, 3, d_model)
        fusion_out = self.transformer_layer(fusion_seq, training=training)
        # 세 토큰에 대해 평균(pooling)하여 단일 feature vector로 결합
        fusion_feature = tf.reduce_mean(fusion_out, axis=1)  # (batch, d_model)
        logits = self.classifier(fusion_feature)
        return logits

# =============================================================================
# 4. 최종 멀티모달 감정 인식 모델 (Inter-Subject)
# =============================================================================
class MultiModalEmotionModel(Model):
    def __init__(self, d_model=64, eeg_n_layers=6, eeg_n_heads=8, eeg_d_ff=2048, eeg_dropout=0.5,
                 fusion_n_heads=4, fusion_d_ff=256, fusion_dropout=0.1, num_subjects=10):
        """
        EEG, Eye, 그리고 subject id를 입력받아 특징을 추출 후 융합하여 감정을 분류합니다.
        num_subjects: 전체 피험자 수 (inter-subject setting)
        """
        super(MultiModalEmotionModel, self).__init__()
        self.eeg_extractor = EEGTransformerEncoder(d_model, eeg_n_layers, eeg_n_heads, eeg_d_ff, eeg_dropout)
        self.eye_extractor = EyeFeatureExtractor()
        self.fusion_module = FusionModule(d_model, fusion_n_heads, fusion_d_ff, fusion_dropout, num_subjects)
    
    def call(self, eeg, eye, subject_ids, training=False):
        eeg_features = self.eeg_extractor(eeg, training=training)
        eye_features = self.eye_extractor(eye, training=training)
        logits = self.fusion_module(eeg_features, eye_features, subject_ids, training=training)
        return logits

# # =============================================================================
# # 5. 데이터 로드 및 전처리
# # =============================================================================

# # -------------------------------
# # 5-0. 전역 상수 및 감정 라벨 매핑
# # -------------------------------
# EEG_DATA_PATH = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_de_features_2D_mapping"
# EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
# EMOTION_MAPPING = {
#     "Excited": 0,
#     "Relaxed": 1,
#     "Stressed": 2,
#     "Bored": 3,
#     "Neutral": 4
# }

# # -------------------------------
# # 5-1. EEG 데이터 로드 함수 (2D mapping DE feature)
# # -------------------------------
# def load_subject_eeg_data(subject, data_path=EEG_DATA_PATH):
#     """
#     주어진 subject(예: "s10")에 대해, data_path 내의 파일명 패턴에 따라 EEG 데이터를 로드합니다.
#     파일명 예: "s10_sample_XX_segment_003_label_Excited_2D.npy"
#     세그먼트 "000", "001", "002"는 건너뛰고, 나머지에 대해 채널 차원을 추가하여 (4,6,6,1) 모양으로 만듭니다.
#     """
#     data, labels = [], []
#     # 파일명을 정렬하여 일관된 순서로 로드
#     files = sorted([f for f in os.listdir(data_path) if f.startswith(subject) and f.endswith("_2D.npy")])
#     for file_name in files:
#         parts = file_name.split("_")
#         if len(parts) < 7:
#             print(f"Unexpected file format: {file_name}")
#             continue
#         segment_name = parts[4]
#         if segment_name in ["000", "001", "002"]:
#             continue  # 건너뛰기
#         emotion_label = parts[-2]
#         if emotion_label in EMOTION_MAPPING:
#             label = EMOTION_MAPPING[emotion_label]
#             file_path = os.path.join(data_path, file_name)
#             try:
#                 de_features = np.load(file_path)  # (4,6,6)
#                 de_features = np.expand_dims(de_features, axis=-1)  # (4,6,6,1)
#                 data.append(de_features)
#                 labels.append(label)
#             except Exception as e:
#                 print(f"Error loading EEG file {file_path}: {e}")
#         else:
#             print(f"Unknown emotion label: {emotion_label} in file {file_name}")
#     if len(data) == 0:
#         print(f"Warning: No EEG data found for {subject}")
#     return np.array(data), np.array(labels)

# # -------------------------------
# # 5-2. Eye Crop 데이터 전처리 함수들
# # -------------------------------
# def downsample_eye_frame(frame):
#     """
#     원본 eye frame 이미지를 다운샘플링합니다.
#     cv2.resize의 dsize 인자는 (width, height) 순서이므로, 여기서는 (32, 8)로 축소합니다.
#     """
#     return cv2.resize(frame, (32, 8), interpolation=cv2.INTER_AREA)

# def reshape_eye_frame(data):
#     """
#     입력 데이터가 4D (N, 32, 64, 3)인 경우, N개 프레임에 대해 평균을 내어 (32,64,3)으로 만든 후,
#     downsample_eye_frame()을 적용합니다.
#     만약 입력이 (32,64,3)이면 그대로 downsample_eye_frame() 적용합니다.
#     """
#     if len(data.shape) == 4 and data.shape[0] > 0:
#         reshaped_data = np.mean(data, axis=0)  # (32,64,3)
#         return downsample_eye_frame(reshaped_data)  # 결과: (8,32,3)
#     elif len(data.shape) == 3 and data.shape == (32, 64, 3):
#         return downsample_eye_frame(data)
#     else:
#         raise ValueError(f"Unexpected eye frame shape: {data.shape}, expected (N,32,64,3) or (32,64,3)")

# # -------------------------------
# # 5-3. 각 피험자별 Eye Crop 데이터 로드 함수
# # -------------------------------
# def load_subject_eye_data(subject, eye_crop_path=EYE_CROP_PATH):
#     """
#     주어진 subject(예: "s10")의 eye crop 데이터를 eye_crop_path 내 subject 폴더에서 로드합니다.
#     파일명은 폴더 내의 모든 .npy 파일을 시간순으로 정렬한 후,  
#     50개씩 그룹으로 묶어 한 샘플(시퀀스)으로 구성합니다.
#     각 프레임은 reshape_eye_frame()으로 전처리한 후,  
#     만약 프레임의 세로 크기가 32라면 (8,32,3) → 패딩을 통해 (8,64,3)로 맞춥니다.
#     최종 샘플 shape: (50, 8, 64, 3)
#     """
#     subject_folder = os.path.join(eye_crop_path, subject)
#     eye_samples = []
#     if not os.path.exists(subject_folder):
#         print(f"Eye crop folder not found for subject: {subject}")
#         return np.array(eye_samples)  # 빈 배열 반환
#     # 파일명을 정렬 (숫자 부분을 기준으로 정렬하는 것을 고려)
#     eye_files = sorted([f for f in os.listdir(subject_folder) if f.endswith(".npy")])
#     # 그룹 단위(50 프레임씩)로 묶기
#     num_chunks = len(eye_files) // 50
#     for i in range(num_chunks):
#         chunk_files = eye_files[i*50:(i+1)*50]
#         frames = []
#         for f in chunk_files:
#             file_path = os.path.join(subject_folder, f)
#             try:
#                 frame_data = np.load(file_path)  # 원본 shape 예: (N,32,64,3) 또는 (32,64,3)
#                 frame_data = reshape_eye_frame(frame_data)  # → (8,32,3)
#             except Exception as e:
#                 print(f"Error processing eye file {file_path}: {e}")
#                 continue
#             # 만약 프레임의 세로 크기가 32이면, (8,32,3) → (8,64,3)로 패딩 적용
#             if frame_data.shape[-2] == 32:
#                 pad_width = [(0, 0)] * frame_data.ndim
#                 pad_width[-2] = (16, 16)  # 양쪽에 16픽셀씩 추가 → 32+16+16 = 64
#                 frame_data = np.pad(frame_data, pad_width, mode='constant', constant_values=0)
#             frames.append(frame_data)
#         # 개선: 50개 미만의 프레임이 있다면 마지막 프레임을 복제하여 50개로 채움
#         if len(frames) > 0 and len(frames) < 50:
#             while len(frames) < 50:
#                 frames.append(frames[-1])
#         if len(frames) == 50:
#             sample_eye = np.stack(frames, axis=0)  # (50, 8, 64, 3)
#             eye_samples.append(sample_eye)
#         else:
#             print(f"Warning: Unable to form a complete sample (50 frames) for subject {subject}. Skipping this sample.")
#     return np.array(eye_samples)

# # -------------------------------
# # 5-4. Inter-subject 멀티모달 데이터 통합 로드 함수
# # -------------------------------
# def load_inter_subject_data(subjects, eeg_data_path=EEG_DATA_PATH, eye_crop_path=EYE_CROP_PATH):
#     """
#     여러 피험자(subjects 목록)에 대해 EEG와 Eye Crop 데이터를 각각 로드하고,
#     subject id와 라벨을 함께 구성합니다.
#     최종 반환 데이터:
#       - eeg_data: (total_samples, 4, 6, 6, 1)
#       - eye_data: (total_samples, 50, 8, 64, 3)
#       - subject_ids: (total_samples,) 정수형
#       - labels: (total_samples,)
#     """
#     all_eeg, all_eye, all_subject_ids, all_labels = [], [], [], []
#     for subject in subjects:
#         # EEG 데이터 로드
#         eeg_data, labels = load_subject_eeg_data(subject, data_path=eeg_data_path)
#         if eeg_data.size == 0:
#             continue
#         # Eye Crop 데이터 로드
#         eye_data = load_subject_eye_data(subject, eye_crop_path=eye_crop_path)
#         if eye_data.size == 0:
#             print(f"Warning: No eye crop data found for {subject}. Creating dummy eye data.")
#             # dummy: 각 EEG 샘플당 (50,8,64,3) 영상을 모두 0으로 채움
#             dummy_eye = np.zeros((eeg_data.shape[0], 50, 8, 64, 3), dtype=np.uint8)
#             eye_data = dummy_eye
#         # EEG와 Eye 데이터 샘플 수 맞추기
#         num_samples = min(eeg_data.shape[0], eye_data.shape[0])
#         if eeg_data.shape[0] != eye_data.shape[0]:
#             print(f"Warning: Mismatch in number of samples for {subject}: EEG={eeg_data.shape[0]}, Eye={eye_data.shape[0]}. Using {num_samples} samples.")
#             eeg_data = eeg_data[:num_samples]
#             eye_data = eye_data[:num_samples]
#             labels = labels[:num_samples]
#         all_eeg.append(eeg_data)
#         all_eye.append(eye_data)
#         # subject id: "sXX"에서 XX를 정수로 변환 (예: "s10" → 10)
#         all_subject_ids.append(np.full((num_samples,), int(subject[1:])))
#         all_labels.append(labels)
#     if len(all_eeg) == 0:
#         raise ValueError("No data loaded for any subject!")
#     # 리스트들을 concatenate
#     all_eeg = np.concatenate(all_eeg, axis=0)
#     all_eye = np.concatenate(all_eye, axis=0)
#     all_subject_ids = np.concatenate(all_subject_ids, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
#     print("최종 EEG data shape:", all_eeg.shape)
#     print("최종 Eye data shape:", all_eye.shape)
#     print("최종 Subject IDs shape:", all_subject_ids.shape)
#     print("최종 Labels shape:", all_labels.shape)
#     return all_eeg, all_eye, all_subject_ids, all_labels

# # -------------------------------
# # 5-5. 전체 데이터 로드 예시 (Inter-Subject)
# # -------------------------------
# # 실제로 subjects 목록을 정의하고 데이터를 로드합니다.
# subjects = [f"s{str(i).zfill(2)}" for i in range(10, 23)]
# eeg_data, eye_data, subject_ids, labels = load_inter_subject_data(subjects, EEG_DATA_PATH, EYE_CROP_PATH)

# # =============================================================================
# # 6. 모델 학습 및 평가
# # =============================================================================
# model = MultiModalEmotionModel(num_subjects=23)  # num_subjects는 실제 피험자 수에 맞게 설정

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=["accuracy"])

# # 모델 학습
# model.fit([eeg_data, eye_data, subject_ids], labels, epochs=10, batch_size=16)

# loss, accuracy = model.evaluate([eeg_data, eye_data, subject_ids], labels)
# print(f"Test Accuracy: {accuracy}")

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


#build_multimodal_data 구현해야함.



def load_multimodal_data(subject):
    eeg_data, eye_data, labels = [], [], []

    eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+)_2D_DE.npy")

    for sample_index in range(40):  # Sample 000 ~ Sample 040까지
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
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
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
        checkpoint_dir = f"/home/bcml1/sigenv/_multimodal_video+eeg/checkpoint/formal_{subject}"
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