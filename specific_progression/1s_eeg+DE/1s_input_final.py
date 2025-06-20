import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Conv3D, Dense, Flatten, LayerNormalization, Dropout, GlobalAveragePooling3D, Reshape
from tensorflow.keras.models import Model

#메모리 문제 해결 위해 아래의 코드를 선언
from tensorflow.keras import mixed_precision
import gc
from tensorflow.keras import backend as K

# 데이터 경로 설정
DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG/2D_1_seg_conv_ch_BPF"
SAVE_PATH = "/home/bcml1/sigenv/1s_eeg+DE/1s"
mixed_precision.set_global_policy('mixed_float16')#메모리 문제 해결하기 위해 선언(+ CNN 모델 크기도 잠시 줄여놓음)

# GELU 정의
# class GELU(tf.keras.layers.Layer):
#     def call(self, inputs):
#         return 0.5 * inputs * (1 + tf.tanh(tf.sqrt(2 / tf.constant(np.pi)) * (inputs + 0.044715 * tf.pow(inputs, 3))))

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
        print(f"Spatial features shape: {spatial_features.shape}")
        spectral_features = self.spectral_conv(inputs)
        print(f"Spectral features shape: {spectral_features.shape}")
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
        print(f"Spatial mask shape: {spatial_mask.shape}")
        spectral_mask = self.spectral_dense(tf.expand_dims(self.spectral_squeeze(inputs), axis=-1))
        print(f"Spectral mask shape: {spectral_mask.shape}")

        spatial_mask = tf.reshape(spatial_mask, [tf.shape(inputs)[0], 1, 1, 1, tf.shape(inputs)[-1]])
        spectral_mask = tf.reshape(spectral_mask, [tf.shape(inputs)[0], 1, 1, 1, tf.shape(inputs)[-1]])

        attended_output = inputs * spatial_mask * spectral_mask
        print(f"Attended output shape: {attended_output.shape}")
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
        print(f"Attention output shape: {attn_output.shape}")
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        print(f"Feedforward output shape: {ffn_output.shape}")
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
        print(f"After conv_block1: {x.shape}")
        x = self.conv_block2(x)
        print(f"After conv_block2: {x.shape}")
        x = self.conv_block3(x)
        print(f"After conv_block3: {x.shape}")
        x = self.attention(x)
        print(f"After attention: {x.shape}")
        x = self.flatten(x)
        print(f"After flatten: {x.shape}")
        x = self.dense_projection(x)
        print(f"After dense projection: {x.shape}")

        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, training=training)
            print(f"After encoder layer {i+1}: {x.shape}")

        return tf.squeeze(self.output_dense(x), axis=1) if len(self.output_dense(x).shape) == 3 else self.output_dense(x)

# 데이터 로드 함수
def load_subject_data(subject):
    data, labels = [], []
    for label, segment_info in enumerate(["negative", "neutral", "positive"]):
        for segment_idx in range(60):
            for sample_idx in range(40):
                file_prefix = f"{subject}_{segment_info}_segment_{segment_idx}_sample_{sample_idx}_2DE.npy"
                file_path = os.path.join(DATA_PATH, file_prefix)
                if os.path.exists(file_path):
                    data.append(np.load(file_path))
                    labels.append(label)
    data_array = np.array(data)
    labels_array = np.array(labels)
    
    if data_array.size == 0:
        print(f"Warning: No data found for {subject}")
    else:
        print(f"Loaded data shape for {subject}: {data_array.shape}")
    return np.array(data), np.array(labels)

# 피험자별 데이터 학습 및 평가
def train_subjects_with_kfold():
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 30)]

    for subject in subjects:
        print(f"Training subject: {subject}")
        subject_data, subject_labels = load_subject_data(subject)
        if len(subject_data) == 0:
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

            # 수정된 코드
            train_dataset = tf.data.Dataset.from_tensor_slices((kfold_train_data, kfold_train_labels)).batch(8).shuffle(1000)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(8)

            #model = TransformerEncoder(input_dim=train_data.shape[-1], n_layers=6, n_heads=8, d_ff=2048, p_drop=0.1, d_model=64)

            # Transformer Encoder 초기화 부분 수정
            model = TransformerEncoder(
                input_dim=train_data.shape[-1],
                n_layers=2,       # 레이어 수 감소 (예: 6 -> 2)
                n_heads=4,        # 헤드 수 감소 (예: 8 -> 4)
                d_ff=512,         # 피드포워드 차원 감소 (예: 2048 -> 512)
                p_drop=0.1,
                d_model=64
            )
            
            # 모델 컴파일 수정
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)

            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(train_dataset, epochs=50, validation_data=val_dataset)

            subject_save_path = os.path.join(SAVE_PATH, f"{subject}_fold_{fold}")
            os.makedirs(subject_save_path, exist_ok=True)

            model.save(os.path.join(subject_save_path, "model.h5"))

            predictions = model.predict(val_dataset)
            predicted_labels = tf.argmax(predictions, axis=-1).numpy()

            report = classification_report(val_labels, predicted_labels, target_names=["Negative", "Neutral", "Positive"], labels=[0, 2], zero_division=0)
            print(report)

            with open(os.path.join(subject_save_path, "classification_report_fold.txt"), "w") as f:
                f.write(report)

            fold += 1

        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
        predictions = model.predict(test_dataset)
        predicted_labels = tf.argmax(predictions, axis=-1).numpy()

        test_report = classification_report(test_labels, predicted_labels, target_names=["Negative", "Neutral", "Positive"], labels=[0, 2], zero_division=0)
        print(f"\nTest Report for {subject}")
        print(test_report)

        with open(os.path.join(SAVE_PATH, f"{subject}_test_report.txt"), "w") as f:
            f.write(test_report)

# 피험자별 학습 실행
train_subjects_with_kfold()
