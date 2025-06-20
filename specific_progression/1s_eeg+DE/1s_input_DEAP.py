import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Conv3D, Dense, Flatten, LayerNormalization, Dropout, GlobalAveragePooling3D
from tensorflow.keras.models import Model

# 메모리 최적화를 위한 mixed precision 설정
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# 데이터 경로 설정
DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
SAVE_PATH = "/home/bcml1/myenv/DEAP_3D1s_Results"
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
        if len(inputs.shape) == 4:  # Expand dims if channel dimension is missing
            inputs = tf.expand_dims(inputs, axis=-1)
        spatial_features = self.spatial_conv(inputs)
        #print(f"Spatial features shape: {spatial_features.shape}")
        spectral_features = self.spectral_conv(inputs)
        #print(f"Spectral features shape: {spectral_features.shape}")
        return spatial_features + spectral_features

# Spatial and Spectral Attention Branch
class SpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialSpectralAttention, self).__init__()
        self.spatial_squeeze = GlobalAveragePooling3D()
        self.spectral_squeeze = GlobalAveragePooling3D()
        self.spatial_dense = Dense(1, activation="sigmoid")
        self.spectral_dense = Dense(1, activation="sigmoid")

    def call(self, inputs):
        spatial_mask = self.spatial_dense(tf.expand_dims(self.spatial_squeeze(inputs), axis=-1))
        #print(f"Spatial mask shape: {spatial_mask.shape}")
        spectral_mask = self.spectral_dense(tf.expand_dims(self.spectral_squeeze(inputs), axis=-1))
        #print(f"Spectral mask shape: {spectral_mask.shape}")

        spatial_mask = tf.reshape(spatial_mask, [tf.shape(inputs)[0], 1, 1, 1, tf.shape(inputs)[-1]])
        spectral_mask = tf.reshape(spectral_mask, [tf.shape(inputs)[0], 1, 1, 1, tf.shape(inputs)[-1]])

        attended_output = inputs * spatial_mask * spectral_mask
        #print(f"Attended output shape: {attended_output.shape}")
        return attended_output + inputs  # Additive residual connection

# Transformer Encoder Layer
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
        if len(inputs.shape) == 2:  # If (batch, features), expand to (batch, 1, features)
            inputs = tf.expand_dims(inputs, axis=1)

        attn_output = self.mha(inputs, inputs, training=training)
        #print(f"Attention output shape: {attn_output.shape}")
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        #print(f"Feedforward output shape: {ffn_output.shape}")
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)

# Transformer Encoder 모델
class TransformerEncoder(tf.keras.Model):
    def __init__(self, input_dim, n_layers=6, n_heads=8, d_ff=2048, p_drop=0.5, d_model=64):
        super(TransformerEncoder, self).__init__()

        # Conv3D 필터 수 감소
        self.conv_block1 = SpatialSpectralConvModule(8, kernel_size=(1, 3, 3), strides=(1, 3, 3))   # 16 -> 8
        self.conv_block2 = SpatialSpectralConvModule(16, kernel_size=(4, 1, 1), strides=(4, 1, 1))  # 32 -> 16
        self.conv_block3 = SpatialSpectralConvModule(32, kernel_size=(1, 2, 2), strides=(1, 2, 2))  # 64 -> 32
#        self.conv_block1 = SpatialSpectralConvModule(16, kernel_size=(3, 3, 3), strides=(1, 2, 2))
#        self.conv_block2 = SpatialSpectralConvModule(32, kernel_size=(3, 3, 3), strides=(1, 2, 2))
#        self.conv_block3 = SpatialSpectralConvModule(64, kernel_size=(3, 3, 3), strides=(1, 2, 2))
        self.attention = SpatialSpectralAttention()
        self.flatten = Flatten()
        self.dense_projection = Dense(d_model, activation="relu")
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)
            for _ in range(n_layers)
        ]
        self.output_dense = Dense(3, activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv_block1(inputs)
        #print(f"After conv_block1: {x.shape}")
        x = self.conv_block2(x)
        #print(f"After conv_block2: {x.shape}")
        x = self.conv_block3(x)
        #print(f"After conv_block3: {x.shape}")
        x = self.attention(x)
        #print(f"After attention: {x.shape}")
        x = self.flatten(x)
        #print(f"After flatten: {x.shape}")
        x = self.dense_projection(x)
        #print(f"After dense projection: {x.shape}")

        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, training=training)
            #print(f"After encoder layer {i+1}: {x.shape}")

        return tf.squeeze(self.output_dense(x), axis=1) if len(self.output_dense(x).shape) == 3 else self.output_dense(x)

# 데이터 로드 함수
def load_subject_data(subject):
    data, labels = [], []
    for file_name in os.listdir(DATA_PATH):
        if file_name.startswith(subject) and file_name.endswith("_2D_DE.npy"):
            emotion_label = file_name.split("_")[-3]
            if emotion_label in EMOTION_MAPPING:
                label = EMOTION_MAPPING[emotion_label]
                file_path = os.path.join(DATA_PATH, file_name)
                data.append(np.load(file_path))
                labels.append(label)
    
    data_array = np.array(data)
    labels_array = np.array(labels)

    # 라벨 검증 추가
    if len(labels_array) > 0:
        unique_labels, label_counts = np.unique(labels_array, return_counts=True)
        print(f"Subject: {subject}")
        print("Unique labels:", unique_labels)
        print("Label counts:", label_counts)

        # 라벨 값이 예상 범위(0~4)에 있는지 확인
        if not all(label in [0, 1, 2, 3, 4] for label in unique_labels):
            print(f"Warning: Labels for {subject} contain unexpected values!")

    return data_array, labels_array

# 학습 및 평가
def train_subjects_with_kfold():
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 32)]

    for subject in subjects:
        print(f"Training subject: {subject}")
        subject_data, subject_labels = load_subject_data(subject)
        subject_data = np.nan_to_num(subject_data, nan=0.0, posinf=0.0, neginf=0.0)
        print("Labels dtype:", subject_labels.dtype)
        subject_labels = subject_labels.astype(np.int32)


        if len(subject_data) == 0:
            print(f"No data found for {subject}. Skipping...")
            continue

        train_data, test_data, train_labels, test_labels = train_test_split(
            subject_data, subject_labels, test_size=0.2, random_state=42
        )

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold = 1

        for train_index, val_index in kf.split(train_data):
            print(f"\nFold {fold} for subject {subject}")

            kfold_train_data, val_data = train_data[train_index], train_data[val_index]
            kfold_train_labels, val_labels = train_labels[train_index], train_labels[val_index]

            # TensorFlow Dataset 생성
            train_dataset = tf.data.Dataset.from_tensor_slices((kfold_train_data, kfold_train_labels)).batch(8).shuffle(1000)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(8)

            # Transformer Encoder 초기화
            model = TransformerEncoder(
                input_dim=train_data.shape[-1],
                n_layers=2,       # 레이어 수
                n_heads=4,        # 헤드 수
                d_ff=512,         # 피드포워드 차원
                p_drop=0.1,
                d_model=64
            )
            
            # Mixed Precision Optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
            
            # 모델 컴파일
            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            # 모델 학습
            model.fit(train_dataset, epochs=50, validation_data=val_dataset)

            # 폴드별 결과 저장 경로 설정
            subject_save_path = os.path.join(SAVE_PATH, f"{subject}_fold_{fold}")
            os.makedirs(subject_save_path, exist_ok=True)

            # 모델 저장
            model.save(os.path.join(subject_save_path, "model.h5"))

            # 검증 데이터셋에 대한 예측
            predictions = model.predict(val_dataset)
            print("Predictions NaN Check:", np.any(np.isnan(predictions)))
            predicted_labels = tf.argmax(predictions, axis=-1).numpy()

            # 검증 데이터셋 결과 레포트
            report = classification_report(
                val_labels,
                predicted_labels,
                target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"],
                labels=[0, 1, 2, 3, 4],
                zero_division=0
            )
            print(report)

            # 레포트 저장
            with open(os.path.join(subject_save_path, "classification_report_fold.txt"), "w") as f:
                f.write(report)

            fold += 1

        # 테스트 데이터셋에 대한 평가
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
        predictions = model.predict(test_dataset)
        predicted_labels = tf.argmax(predictions, axis=-1).numpy()

        # 테스트 데이터셋 결과 레포트
        test_report = classification_report(
            test_labels,
            predicted_labels,
            target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"],
            labels=[0, 1, 2, 3, 4],
            zero_division=0
        )
        print(f"\nTest Report for {subject}")
        print(test_report)

        # 테스트 레포트 저장
        with open(os.path.join(SAVE_PATH, f"{subject}_test_report.txt"), "w") as f:
            f.write(test_report)

# 실행
train_subjects_with_kfold()
