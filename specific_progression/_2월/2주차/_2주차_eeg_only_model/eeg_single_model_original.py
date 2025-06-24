import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv3D, Dense, Flatten, LayerNormalization, Dropout, GlobalAveragePooling3D
from tensorflow.keras.models import Model

# # 메모리 최적화를 위한 mixed precision 설정
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")

#메모리 제한
def limit_gpu_memory(memory_limit_mib=15000):
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

# 적용
limit_gpu_memory(15000)

# 데이터 경로 설정
DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap_labeled"
SAVE_PATH = "/home/bcml1/myenv/DEAP_EEG_Original"
os.makedirs(SAVE_PATH, exist_ok=True)

# 감정 라벨 매핑
EMOTION_MAPPING = {
    "Negative": 0,
    "Positive": 1,
    "Neutral": 2,
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

# def average_model_weights(model, model_paths):
    """
    Average the weights of models stored in model_paths and assign them to the model.

    Args:
        model (tf.keras.Model): The model to assign the averaged weights.
        model_paths (list): List of file paths to the saved model files.

    Returns:
        tf.keras.Model: The model with averaged weights.
    """
    # Register custom objects
    # custom_objects = {
    #     "TransformerEncoder": TransformerEncoder,
    #     "SpatialSpectralConvModule": SpatialSpectralConvModule,
    #     "SpatialSpectralAttention": SpatialSpectralAttention,
    #     "TransformerEncoderLayer": TransformerEncoderLayer,
    # }
    # weights = []
    # for path in model_paths:
    #     try:
    #         # 모델 초기화 및 가중치 로드
    #         temp_model = TransformerEncoder(
    #             input_dim=model.input_dim,
    #             n_layers=model.n_layers,
    #             n_heads=model.n_heads,
    #             d_ff=model.d_ff,
    #             p_drop=model.p_drop,
    #             d_model=model.d_model
    #         )
    #         temp_model.load_weights(path)
    #         weights.append(temp_model.get_weights())
    #     except Exception as e:
    #         print(f"Error loading weights from {path}: {e}")
    #         continue

    # if len(weights) == 0:
    #     raise ValueError("No valid models were loaded. Cannot average weights.")

    # # 평균 가중치 계산
    # avg_weights = [np.mean([weight[i] for weight in weights], axis=0) for i in range(len(weights[0]))]
    # model.set_weights(avg_weights)
    # return model
def average_model_weights(model, model_paths):
    weights = []
    for path in model_paths:
        try:
            temp_model = TransformerEncoder(
                input_dim=model.input_dim,
                n_layers=model.n_layers,
                n_heads=model.n_heads,
                d_ff=model.d_ff,
                p_drop=model.p_drop,
                d_model=model.d_model
            )
            temp_model.build((None, model.input_dim))  # 모델을 먼저 빌드
            temp_model.load_weights(path)
            weights.append(temp_model.get_weights())
        except Exception as e:
            print(f"Error loading weights from {path}: {e}")
            continue

    if len(weights) == 0:
        raise ValueError("No valid models were loaded. Cannot average weights.")

    avg_weights = [np.mean([weight[i] for weight in weights], axis=0) for i in range(len(weights[0]))]
    model.set_weights(avg_weights)
    return model


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
    
    def get_config(self):
        return {}
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
    
    def get_config(self):
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
        }

# Transformer Encoder 모델
class TransformerEncoder(tf.keras.Model):
    def __init__(self, input_dim, n_layers=6, n_heads=8, d_ff=2048, p_drop=0.5, d_model=64):
        super(TransformerEncoder, self).__init__()
        # 인스턴스 변수 초기화
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.d_model = d_model

        # 모델 구성 요소 정의
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

    def build(self, input_shape):
        """ 모델을 명확하게 초기화 """
        super(TransformerEncoder, self).build(input_shape)

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

    def get_config(self):
        # 인스턴스 변수의 값을 반환
        return {
            "input_dim": self.input_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "p_drop": self.p_drop,
            "d_model": self.d_model,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
 
# 데이터 로드 함수
def load_subject_data(subject):
    data, labels = [], []
    for file_name in os.listdir(DATA_PATH):
        if file_name.startswith(subject) and file_name.endswith(".npy"):
            # 파일명에서 segment 이름 추출 (예: segment_00)
            segment_name = file_name.split("_")[4]
            # if segment_name in ["000", "001", "002"]:
            #     print(f"skipping segment: {segment_name}")
            #     continue  # 해당 segment 건너뜀
            
            # 파일명에서 감정 라벨 추출 후 3개 클래스로 매핑
            # (예: 파일명에 포함된 "Excited", "Relaxed", "Stressed", "Bored", "Neutral")
            emotion_label = file_name.split("_")[-1]
            emotion_label = emotion_label.replace(".npy", "")
            if emotion_label in EMOTION_MAPPING:
                label = EMOTION_MAPPING[emotion_label]
                file_path = os.path.join(DATA_PATH, file_name)
                data.append(np.load(file_path))
                labels.append(label)
    
    data_array = np.array(data)
    labels_array = np.array(labels)

    if len(labels_array) > 0:
        unique_labels, label_counts = np.unique(labels_array, return_counts=True)
        print(f"Subject: {subject}")
        print("Unique labels:", unique_labels)
        print("Label counts:", label_counts)
        # 3개 클래스 확인: 0, 1, 2
        if not all(label in [0, 1, 2] for label in unique_labels):
            print(f"Warning: Labels for {subject} contain unexpected values!")

    return data_array, labels_array

   

# 학습 및 평가
# def train_subjects_with_kfold():
#     subjects = [f"s{str(i).zfill(2)}" for i in range(1, 32)]
#     final_model_path = os.path.join(SAVE_PATH, "final_model")
#     os.makedirs(final_model_path, exist_ok=True)


#     for subject in subjects:
#         print(f"Training subject: {subject}")
#         subject_data, subject_labels = load_subject_data(subject)
#         subject_data = np.nan_to_num(subject_data, nan=0.0, posinf=0.0, neginf=0.0)

#         # 라벨 값의 범위를 확인 및 보정
#         assert subject_labels.max() < 5, f"Labels contain invalid value: {subject_labels.max()}"
#         valid_labels = [0, 1, 2, 3, 4]
#         valid_indices = np.isin(subject_labels, valid_labels)
#         subject_data = subject_data[valid_indices]
#         subject_labels = subject_labels[valid_indices]

#         print("Labels dtype:", subject_labels.dtype)
#         subject_labels = subject_labels.astype(np.int32)

#         if len(subject_data) == 0:
#             print(f"No data found for {subject}. Skipping...")
#             continue

#         train_data, test_data, train_labels, test_labels = train_test_split(
#             subject_data, subject_labels, test_size=0.2, random_state=42
#         )

#         kf = KFold(n_splits=10, shuffle=True, random_state=42)
#         fold = 1
#         fold_model_paths = []  # Store paths to each fold's model

#         for train_index, val_index in kf.split(train_data):
#             print(f"\nFold {fold} for subject {subject}")

#             kfold_train_data, val_data = train_data[train_index], train_data[val_index]
#             kfold_train_labels, val_labels = train_labels[train_index], train_labels[val_index]

#             # TensorFlow Dataset 생성
#             train_dataset = tf.data.Dataset.from_tensor_slices((kfold_train_data, kfold_train_labels)).batch(8).shuffle(1000)
#             val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(8)

#             # 모델 초기화 및 학습
#             model = TransformerEncoder(
#                 input_dim=train_data.shape[-1],
#                 n_layers=2,       # 레이어 수
#                 n_heads=4,        # 헤드 수
#                 d_ff=512,         # 피드포워드 차원
#                 p_drop=0.1,
#                 d_model=64
#             )
            
#             lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#                 initial_learning_rate=1e-4,
#                 decay_steps=10000,
#                 decay_rate=0.9
#             )   
#             # Mixed Precision Optimizer
#             optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#             optimizer = mixed_precision.LossScaleOptimizer(optimizer)
            
#             # Class weight 계산
#             # unique_classes = np.unique(train_labels)
#             # class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
#             # class_weight_dict = dict(enumerate(class_weights))

            
#             # 모델 컴파일
#             model.compile(
#                 optimizer=optimizer,
#                 loss="sparse_categorical_crossentropy",
#                 metrics=["accuracy"]
#             )
#             print("Model compiled successfully.")

#             debug_model_outputs(model, train_dataset)
#             debug_loss(model, train_dataset)


#             # 모델 학습
#             model.fit(
#                 train_dataset, 
#                 epochs=50, 
#                 validation_data=val_dataset,
#                 # class_weight=class_weight_dict    
#             )

#             # 폴드별 결과 저장 경로 설정
#             subject_save_path = os.path.join(SAVE_PATH, f"{subject}_fold_{fold}")
#             os.makedirs(subject_save_path, exist_ok=True)

#             # 모델 저장
#             fold_model_path = os.path.join(subject_save_path, "model.h5")
#             model.save(fold_model_path)
#             fold_model_paths.append(fold_model_path)

#             # 검증 데이터셋에 대한 예측
#             predictions = model.predict(val_dataset)
#             print("Predictions NaN Check:", np.any(np.isnan(predictions)))
#             predicted_labels = tf.argmax(predictions, axis=-1).numpy()

#             # 검증 데이터셋 결과 레포트
#             report = classification_report(
#                 val_labels,
#                 predicted_labels,
#                 target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"],
#                 labels=[0, 1, 2, 3, 4],
#                 zero_division=0
#             )
#             print(report)

#             # 레포트 저장
#             with open(os.path.join(subject_save_path, "classification_report_fold.txt"), "w") as f:
#                 f.write(report)

#             fold += 1

#         # 가중치 평균화를 통한 최종 모델 생성
#         final_model = TransformerEncoder(
#             input_dim=train_data.shape[-1],
#             n_layers=2,
#             n_heads=4,
#             d_ff=512,
#             p_drop=0.1,
#             d_model=64
#         )
#         final_model.compile(
#             optimizer=optimizer,
#             loss="sparse_categorical_crossentropy",
#             metrics=["accuracy"]
#         )
#         final_model = average_model_weights(final_model, fold_model_paths)

#         # 최종 모델을 전체 train 데이터로 추가 학습
#         full_train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(8).shuffle(1000)
#         final_model.fit(
#             full_train_dataset,
#             epochs=50
#         )

#         # 최종 모델 저장
#         final_model.save(os.path.join(final_model_path, "model.h5"))
#         print(f"Final model saved for subject {subject}.")


#         # 테스트 데이터셋에 대한 평가
#         test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
#         predictions = final_model.predict(test_dataset)
#         predicted_labels = tf.argmax(predictions, axis=-1).numpy()

#         # 테스트 데이터셋 결과 레포트
#         test_report = classification_report(
#             test_labels,
#             predicted_labels,
#             target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"],
#             labels=[0, 1, 2, 3, 4],
#             zero_division=0
#         )
#         print(f"\nTest Report for {subject}")
#         print(test_report)

#         # 테스트 레포트 저장
#         with open(os.path.join(SAVE_PATH, f"{subject}_test_report.txt"), "w") as f:
#             f.write(test_report)
def train_subjects_with_kfold():
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    final_model_path = os.path.join(SAVE_PATH, "final_model")
    os.makedirs(final_model_path, exist_ok=True)

    for subject in subjects:
        print(f"Training subject: {subject}")
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        subject_data, subject_labels = load_subject_data(subject)
        # NaN 및 Inf 처리
        subject_data = np.nan_to_num(subject_data, nan=0.0, posinf=0.0, neginf=0.0)

        # 유효 라벨 필터링 (0, 1, 2)
        valid_labels = [0, 1, 2]
        valid_indices = np.isin(subject_labels, valid_labels)
        subject_data = subject_data[valid_indices]
        subject_labels = subject_labels[valid_indices].astype(np.int32)

        if len(subject_data) == 0:
            print(f"No data found for {subject}. Skipping...")
            continue

        # train/test 분리
        train_data, test_data, train_labels, test_labels = train_test_split(
            subject_data, subject_labels, test_size=0.2, random_state=42, stratify=subject_labels
        )

        # K-Fold (10-fold)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_model_paths = []

        for fold, (train_index, val_index) in enumerate(kf.split(train_data, train_labels)):
            print(f"\nFold {fold+1} for subject {subject}")

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_data[train_index], train_labels[train_index])
            ).batch(8).shuffle(1000)
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (train_data[val_index], train_labels[val_index])
            ).batch(8)

            # 모델 생성 (입력 shape: (32, 1280))
            model = TransformerEncoder(
                input_dim=train_data.shape[-1],
                n_layers=2,
                n_heads=4,
                d_ff=512,
                p_drop=0.1,
                d_model=128
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            debug_model_outputs(model, train_dataset)
            debug_loss(model, train_dataset)

            model.fit(train_dataset, epochs=50, validation_data=val_dataset)

            # 폴드별 가중치 저장 (가중치만 저장)
            fold_model_path = os.path.join(subject_save_path, f"{subject}_fold_{fold+1}.weights.h5")
            model.save_weights(fold_model_path)
            fold_model_paths.append(fold_model_path)

        # 평균 가중치를 적용하여 최종 모델 생성
        final_model = TransformerEncoder(
            input_dim=train_data.shape[-1],
            n_layers=2,
            n_heads=4,
            d_ff=512,
            p_drop=0.1,
            d_model=128
        )
        final_model.build((None, train_data.shape[1], train_data.shape[2]))
        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        final_model = average_model_weights(final_model, fold_model_paths)

        # 전체 train 데이터로 추가 학습
        full_train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(8).shuffle(1000)
        final_model.fit(full_train_dataset, epochs=50)

        # 최종 모델 가중치 저장
        final_model.save_weights(os.path.join(subject_save_path, "final_model.weights.h5"))
        print(f"Final model weights saved for subject {subject}.")

        # 테스트 데이터 평가
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
        predictions = final_model.predict(test_dataset)
        predicted_labels = tf.argmax(predictions, axis=-1).numpy()

        test_report = classification_report(
            test_labels, predicted_labels, 
            target_names=["negative", "positive", "neutral"],
            labels=[0, 1, 2], zero_division=0
        )
        print(f"\nTest Report for {subject}")
        print(test_report)

        with open(os.path.join(final_model_path, f"{subject}_test_report.txt"), "w") as f:
            f.write(test_report)

# 실행
train_subjects_with_kfold()
