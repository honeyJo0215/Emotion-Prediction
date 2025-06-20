import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
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

def debug_loss(model, dataset):
    for data, labels in dataset.take(1):
        outputs = model(data, training=True)
        print(f"Model outputs: {outputs.numpy()}")
        print(f"Labels: {labels.numpy()}")

def debug_model_outputs(model, dataset):
    for data, labels in dataset.take(1):
        outputs = model(data, training=True)
        print(f"Model output NaN: {np.isnan(outputs.numpy()).any()}, Inf: {np.isinf(outputs.numpy()).any()}")
        print(f"Model output shape: {outputs.shape}, Labels shape: {labels.shape}")


# 데이터 검증 및 전처리 함수
def preprocess_data(data):
    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)  # NaN 값을 0으로 대체
    data = tf.where(tf.math.is_inf(data), tf.zeros_like(data), data)  # Inf 값을 0으로 대체
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)
    range_val = max_val - min_val + 1e-8  # 분모가 0이 되지 않도록 작은 값을 추가
    data = (data - min_val) / range_val
    return data


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

# Spatial and Spectral Attention Branch
class SpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialSpectralAttention, self).__init__()
        self.spatial_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")
        self.spectral_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")

    def call(self, inputs):
        if len(inputs.shape) == 4:  # (batch_size, depth, height, width)
            inputs = tf.expand_dims(inputs, axis=-1)  # (batch_size, depth, height, width, channels)
    
        # Spatial attention
        spatial_mask = self.spatial_squeeze(inputs)
        spatial_output = inputs * spatial_mask
        #print(f"Spatial mask shape: {spatial_mask.shape}, Spatial output shape: {spatial_output.shape}")

        # Spectral attention
        spectral_mask = self.spectral_squeeze(inputs)
        spectral_output = inputs * spectral_mask
        #print(f"Spectral mask shape: {spectral_mask.shape}, Spectral output shape: {spectral_output.shape}")

        # Combine spatial and spectral outputs
        combined_output = spatial_output + spectral_output
        print(f"Combined output shape: {combined_output.shape}")
        return combined_output
    
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
        if len(inputs.shape) == 2:  # (batch_size, features)
            inputs = tf.expand_dims(inputs, axis=1)  # (batch_size, 1, features)
    
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Transformer Encoder 모델
class TransformerEncoder(tf.keras.Model):
    def __init__(self, input_dim, n_layers=6, n_heads=8, d_ff=2048, p_drop=0.5, d_model=64):
        super(TransformerEncoder, self).__init__()
        self.conv_block1 = SpatialSpectralConvModule(8, kernel_size=(1, 3, 3), strides=(1, 3, 3))
        self.conv_block2 = SpatialSpectralConvModule(16, kernel_size=(4, 1, 1), strides=(4, 1, 1))
        self.conv_block3 = SpatialSpectralConvModule(32, kernel_size=(1, 2, 2), strides=(1, 2, 2))
        self.attention = SpatialSpectralAttention()
        self.flatten = Flatten()
        self.dense_projection = Dense(d_model, activation="relu")
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)
            for _ in range(n_layers)
        ]
        self.output_dense = Dense(5, activation="softmax")

    def call(self, inputs, training=False):
        x = preprocess_data(inputs)
        x = self.attention(x)  # Apply attention first
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense_projection(x)
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        return tf.squeeze(self.output_dense(x), axis=1)
    
# 데이터 로드 함수
def load_subject_data(subject):
    data, labels = [], []
    for file_name in os.listdir(DATA_PATH):
        if file_name.startswith(subject) and file_name.endswith("_2D_DE.npy"):
            emotion_label = file_name.split("_")[-3]
            print(f"Emotion labels:{emotion_label}")
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

        # 라벨 값의 범위를 확인 및 보정
        assert subject_labels.max() < 5, f"Labels contain invalid value: {subject_labels.max()}"
        valid_labels = [0, 1, 2, 3, 4]
        valid_indices = np.isin(subject_labels, valid_labels)
        subject_data = subject_data[valid_indices]
        subject_labels = subject_labels[valid_indices]

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

            # 모델 초기화 및 학습
            model = TransformerEncoder(
                input_dim=train_data.shape[-1],
                n_layers=2,       # 레이어 수
                n_heads=4,        # 헤드 수
                d_ff=512,         # 피드포워드 차원
                p_drop=0.1,
                d_model=64
            )
            
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4,
                decay_steps=10000,
                decay_rate=0.9
            )   
            # Mixed Precision Optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
            
            # Class weight 계산
            # unique_classes = np.unique(train_labels)
            # class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
            # class_weight_dict = dict(enumerate(class_weights))

            
            # 모델 컴파일
            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            print("Model compiled successfully.")

            debug_model_outputs(model, train_dataset)
            debug_loss(model, train_dataset)


            # 모델 학습
            model.fit(
                train_dataset, 
                epochs=50, 
                validation_data=val_dataset,
                # class_weight=class_weight_dict    
            )

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
