import os
import re
import gc
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Reshape
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Conv3D, BatchNormalization, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# GPU 메모리 제한 (필요 시)
def limit_gpu_memory(memory_limit_mib=5000):
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

limit_gpu_memory(5000)


# 데이터 경로 설정
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
SAVE_PATH = "/home/bcml1/myenv/DEAP_Multimodal_cross"
os.makedirs(SAVE_PATH, exist_ok=True)

# 감정 라벨 매핑
EMOTION_MAPPING = {
    "Excited": 0,
    "Relaxed": 1,
    "Stressed": 2,
    "Bored": 3,
    "Neutral": 4
}


# Spatial-Spectral Convolution Module
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


# Spatial and Spectral Attention Branch
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


class EEGCNN(tf.keras.Model):
    def __init__(self, d_model):
        super(EEGCNN, self).__init__()
        self.conv_block1 = SpatialSpectralConvModule(8, kernel_size=(1, 3, 3), strides=(1, 3, 3))
        self.conv_block2 = SpatialSpectralConvModule(16, kernel_size=(4, 1, 1), strides=(4, 1, 1))
        self.conv_block3 = SpatialSpectralConvModule(32, kernel_size=(1, 2, 2), strides=(1, 2, 2))
        self.attention = SpatialSpectralAttention()
        self.flatten = Flatten()
        self.dense_projection = Dense(d_model, activation="relu")

    def call(self, inputs):
        x = self.attention(inputs)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense_projection(x)
        return x
    
# Eye Crop CNN 모델
def build_eye_crop_model(d_model=64):  # d_model 추가
    eye_input = Input(shape=(50, 8, 64, 3))  # 현재 입력 크기
    x = Reshape((50, 8, 64, 3))(eye_input)  # ✅ 올바른 형태로 변환
    #x = Lambda(lambda t: tf.expand_dims(t, axis=1))(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(d_model, activation='relu')(x)  # **d_model=64로 통일**
    return Model(eye_input, x, name="EyeCrop_CNN")

# Transformer Encoder Layer
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


# 🟢 **Cross-Modal Transformer Layer**
class CrossModalTransformer(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(CrossModalTransformer, self).__init__()
        
        # EEG → Eye-tracking, Eye-tracking → EEG 간의 cross-modal attention
        self.eeg_to_eye_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)
        self.eye_to_eeg_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)
        
        # Feedforward network
        self.ffn = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(input_dim)
        ])
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, eeg_input, eye_input):
        # EEG → Eye-tracking cross-modal attention
        eeg_to_eye = self.eeg_to_eye_attention(query=eeg_input, value=eye_input, key=eye_input)
        
        # Eye-tracking → EEG cross-modal attention
        eye_to_eeg = self.eye_to_eeg_attention(query=eye_input, value=eeg_input, key=eeg_input)
        
        # Combine both modalities
        combined = tf.concat([eeg_to_eye, eye_to_eeg], axis=-1)
        
        # Feedforward processing
        combined = self.ffn(combined)
        
        # Apply Layer Normalization
        combined = self.layer_norm(combined)
        
        return combined

# Transformer Encoder Layer
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

# Transformer Encoder
def build_transformer_encoder(input_dim, n_layers=2, n_heads=4, d_ff=512, dropout_rate=0.1, d_model=64, name="Transformer_Encoder"):
    inputs = Input(shape=(input_dim, d_model))
    x = inputs
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate)(x)
    return Model(inputs, x, name=name)


# 🟢 **멀티모달 모델 구성**
def build_multimodal_model(eeg_input_shape, eye_input_shape=(50, 8, 64, 3)):
    eeg_input = Input(shape=eeg_input_shape)
    eye_input = Input(shape=eye_input_shape)

    # EEG CNN
    eeg_cnn_model = EEGCNN(d_model=64)
    eeg_features = eeg_cnn_model(eeg_input)

    # Eye Crop CNN
    eye_cnn_model = build_eye_crop_model(d_model=64)
    eye_features = eye_cnn_model(eye_input)

    expand_dims_layer = Lambda(lambda x: tf.tile(x[:, None, :], [1, 64, 1]))

    # Intra-modality Transformers
    eeg_transformer = build_transformer_encoder(input_dim=64, name="Transformer_Encoder_EEG")
    eye_transformer = build_transformer_encoder(input_dim=64, name="Transformer_Encoder_EYE")  
    eeg_transformed = eeg_transformer(expand_dims_layer(eeg_features))  
    eye_transformed = eye_transformer(expand_dims_layer(eye_features))  

    # 🔥 **Cross-Modal Transformer 적용**
    cross_modal_transformer = CrossModalTransformer(input_dim=64, hidden_dim=128, num_heads=4)
    fused_features = cross_modal_transformer(eeg_transformed, eye_transformed)

    # Classification Head
    x = Dense(128, activation="relu")(fused_features)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(5, activation="softmax")(x)

    output = Lambda(lambda x: tf.reduce_mean(x, axis=1))(output)  

    model = Model(inputs=[eeg_input, eye_input], outputs=output, name="Multimodal_Model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    return model

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

        # 멀티모달 모델 구축
        model = build_multimodal_model(eeg_input_shape=train_eeg.shape[1:])
        
        # 라벨 차원 확장
        train_labels = np.expand_dims(train_labels, axis=-1)
        valid_labels = np.expand_dims(valid_labels, axis=-1)

        # Train + Validation 학습 진행
        model.fit(
            [train_eeg, train_eye], train_labels,
            validation_data=([valid_eeg, valid_eye], valid_labels), 
            epochs=50, batch_size=2
        )

        # 모델 저장
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        model.save_weights(os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5"))

        # 테스트 평가
        predictions = model.predict([test_eeg, test_eye])
        predicted_labels = np.argmax(predictions, axis=-1)

        # Classification Report 저장
        test_report = classification_report(
            test_labels, predicted_labels, 
            target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"], 
            labels=[0, 1, 2, 3, 4],
            zero_division=0
        )

        print(f"\nTest Report for {subject}")
        print(test_report)

        with open(os.path.join(subject_save_path, f"{subject}_test_report.txt"), "w") as f:
            f.write(test_report)

if __name__ == "__main__":
    train_multimodal()
