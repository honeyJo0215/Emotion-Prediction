#이 코드를 수정해서 intra 코드를 완성하자. 참고로 지금까지 eeg_label_DE(1s_intra)_1 코드가 가장 성능이 좋았음...

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv3D, Dense, Flatten, LayerNormalization, Dropout, GlobalAveragePooling3D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # 메모리 최적화를 위한 mixed precision 설정
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")

#메모리 제한
def limit_gpu_memory(memory_limit_mib=5000):
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
limit_gpu_memory(5000)

# 데이터 경로 설정
DATA_PATH = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_de_features_2D_mapping"
SAVE_PATH = "/home/bcml1/sigenv/_1주차_eeg_1s_intra_DEAP/test_result6"
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
def preprocess_data(data, apply_min_max=False):
    # data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)  # NaN 값을 0으로 대체
    # data = tf.where(tf.math.is_inf(data), tf.zeros_like(data), data)  # Inf 값을 0으로 대체
    if apply_min_max:
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
    
    def get_config(self):
        return {
            "filters": self.spatial_conv.filters,
            "kernel_size": self.spatial_conv.kernel_size,
            "strides": self.spatial_conv.strides,
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

        # Spectral attention
        spectral_mask = self.spectral_squeeze(inputs)
        spectral_output = inputs * spectral_mask

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
        if len(inputs.shape) == 2:  # (batch_size, features)
            inputs = tf.expand_dims(inputs, axis=1)  # (batch_size, 1, features)
    
        attn_output = self.mha(inputs, inputs, training=training)    # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)    # Residual connection and normalization
        ffn_output = self.ffn(out1) # Feed-forward network
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection and normalization
        return out2
    
    def get_config(self):
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
        }

# Transformer Encoder 모델
class TransformerEncoder(tf.keras.Model):
    def __init__(self, input_shape, n_layers=6, n_heads=8, d_ff=2048, p_drop=0.5, d_model=64):
        super(TransformerEncoder, self).__init__()
        # 인스턴스 변수 초기화
        # self.input_dim = input_dim
        self.input_shape = input_shape  # 전체 입력 형태 저장
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.d_model = d_model

        # 모델 구성 요소 정의
        self.attention = SpatialSpectralAttention()
        self.conv_block1 = SpatialSpectralConvModule(8, kernel_size=(1, 3, 3), strides=(1, 3, 3))
        self.conv_block2 = SpatialSpectralConvModule(16, kernel_size=(4, 1, 1), strides=(4, 1, 1))
        self.conv_block3 = SpatialSpectralConvModule(32, kernel_size=(1, 2, 2), strides=(1, 2, 2))
        # self.flatten = Flatten()
        self.dense_projection = Dense(d_model, activation="relu")
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)
            for _ in range(n_layers)
        ]
        self.global_pool = GlobalAveragePooling3D()  # 시퀀스 유지
        self.output_dense = Dense(5, activation="softmax")

    def build(self, input_shape):
        """ 모델을 명확하게 초기화 """
        super(TransformerEncoder, self).build(input_shape)

    def call(self, inputs, training=False):
        x = preprocess_data(inputs, apply_min_max=False)  # 이미 표준화 되었으므로 Min-Max 정규화 비활성화
        x = self.attention(x)  # Apply attention first
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # x = self.flatten(x)
        x = self.dense_projection(x)
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        x = self.global_pool(x)  # 시퀀스 유지
        # return tf.squeeze(self.output_dense(x), axis=1)
        return self.output_dense(x)  # (batch_size, 5)

    def get_config(self):
        # 인스턴스 변수의 값을 반환
        return {
            "input_shape": self.input_shape,
            # "input_dim": self.input_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "p_drop": self.p_drop,
            "d_model": self.d_model,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# 데이터 로드 함수 s01_sample_00_segment_000_label_Excited_2D.npy
def load_subject_data(subject, data_path):
    data, labels = [], []
    
    # 감정 라벨 리스트 (Excited, Relaxed, Stressed, Bored, Neutral)
    emotions = ["Excited", "Relaxed", "Stressed", "Bored", "Neutral"]
    
    for label, emotion in enumerate(emotions):  # 0~4의 라벨 할당
        for segment_idx in range(3, 63):  # segment_003~segment_062
            for sample_idx in range(40):  # 40개의 샘플
                # 파일명 일관성 유지
                file_prefix = f"{subject}_sample_{str(sample_idx).zfill(2)}_segment_{str(segment_idx).zfill(3)}_label_{emotion}_2D.npy"
                
                file_path = os.path.join(data_path, file_prefix)

                # 파일이 존재하면 로드
                if os.path.exists(file_path):
                    try:
                        de_features = np.load(file_path)  # 데이터 로드
                        # 채널 축 추가: (4, 6, 6) -> (4, 6, 6, 1)
                        de_features = np.expand_dims(de_features, axis=-1)
                        data.append(de_features)  # 데이터 추가
                        labels.append(label)  # 감정 라벨 추가
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
    
    # 리스트를 NumPy 배열로 변환
    data_array = np.array(data)
    labels_array = np.array(labels)
    
    if data_array.size == 0:
        print(f"Warning: No data found for {subject}")
    else:
        print(f"Loaded data shape for {subject}: {data_array.shape}")

    return data_array, labels_array

# 모델 예측 앙상블 함수
def ensemble_predictions(model_paths, test_dataset, input_shape):
    predictions = []
    for path in model_paths:
        model = TransformerEncoder(input_shape=input_shape, n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
        model.build(input_shape=(None,) + input_shape)  # (batch_size, 4, 6, 6, 1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.load_weights(path)
        preds = model.predict(test_dataset, verbose=0)
        predictions.append(preds)
    avg_preds = np.mean(predictions, axis=0)
    return tf.argmax(avg_preds, axis=-1).numpy()

# def average_model_weights(model, model_paths):
#     weights = []
#     for path in model_paths:
#         try:
#             temp_model = TransformerEncoder(
#                 input_dim=model.input_dim,
#                 n_layers=model.n_layers,
#                 n_heads=model.n_heads,
#                 d_ff=model.d_ff,
#                 p_drop=model.p_drop,
#                 d_model=model.d_model
#             )
#             temp_model.build((None, model.input_dim))  # 모델을 먼저 빌드
#             temp_model.load_weights(path)
#             weights.append(temp_model.get_weights())
#         except Exception as e:
#             print(f"Error loading weights from {path}: {e}")
#             continue

#     if len(weights) == 0:
#         raise ValueError("No valid models were loaded. Cannot average weights.")

#     avg_weights = [np.mean([weight[i] for weight in weights], axis=0) for i in range(len(weights[0]))]
#     model.set_weights(avg_weights)
#     return model

# 학습 및 평가
def train_subjects_with_kfold():
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 32)]
    # final_model_path = os.path.join(SAVE_PATH, "final_model")
    # os.makedirs(final_model_path, exist_ok=True)

    for subject in subjects:
        print(f"\n=== Training subject: {subject} ===")
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        subject_data, subject_labels = load_subject_data(subject, DATA_PATH)
        
        if subject_data.size == 0:
            print(f"No data found for {subject}. Skipping...")
            continue

        # NaN과 Inf 값 처리
        subject_data = np.nan_to_num(subject_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        valid_labels = [0, 1, 2, 3, 4]
        valid_indices = np.isin(subject_labels, valid_labels)
        subject_data = subject_data[valid_indices]
        subject_labels = subject_labels[valid_indices]
        subject_labels = subject_labels.astype(np.int32)

        if len(subject_data) == 0:
            print(f"No data found for {subject}. Skipping...")
            continue

        train_data, test_data, train_labels, test_labels = train_test_split(
            subject_data, subject_labels, test_size=0.2, random_state=42, stratify=subject_labels
        )

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold_model_paths = []

        for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
            print(f"\n--- Fold {fold+1} for subject {subject} ---")

            kfold_train_data, val_data = train_data[train_index], train_data[val_index]
            kfold_train_labels, val_labels = train_labels[train_index], train_labels[val_index]

            # TensorFlow Dataset 생성
            train_dataset = tf.data.Dataset.from_tensor_slices((kfold_train_data, kfold_train_labels))
            train_dataset = train_dataset.shuffle(buffer_size=1000).batch(8)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(8)

            # 모델 초기화 및 학습
            model = TransformerEncoder(
                input_shape=(4, 6, 6, 1),
                n_layers=2,       # 레이어 수
                n_heads=4,        # 헤드 수
                d_ff=512,         # 피드포워드 차원
                p_drop=0.3,
                d_model=64
            )

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            print("Model compiled successfully.")
            
            # 콜백 정의
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

            # 모델 학습
            history = model.fit(
                train_dataset, 
                epochs=50, 
                validation_data=val_dataset,
                callbacks=[lr_scheduler]
            )

            # 폴드별 가중치 저장
            fold_model_path = os.path.join(subject_save_path, f"{subject}_fold_{fold+1}.weights.h5")
            model.save_weights(fold_model_path)
            fold_model_paths.append(fold_model_path)
            print(f"Fold {fold+1} model weights saved to {fold_model_path}")

            # 메모리 관리
            del model
            tf.keras.backend.clear_session()
        
        # 앙상블 예측을 위해 폴드별 모델 경로에 final_model을 추가하지 않습니다.
        
        # 테스트 데이터셋 생성
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
        
        # 앙상블 예측 수행
        input_shape = (4, 6, 6, 1)  # (depth, height, width, channels)
        ensemble_predicted_labels = ensemble_predictions(fold_model_paths, test_dataset, input_shape)
        
        # # 앙상블을 위한 최종 모델 생성 및 가중치 평균화
        # print(f"\n--- Averaging weights for subject {subject} ---")
        # final_model = TransformerEncoder(
        #     input_dim=train_data.shape[-1],
        #     n_layers=2,
        #     n_heads=4,
        #     d_ff=512,
        #     p_drop=0.1,
        #     d_model=64
        # )
        
        # final_model.build((None, train_data.shape[-1]))  # 모델을 먼저 빌드
        # final_model = average_model_weights(final_model, fold_model_paths)
        # # 최종 모델 컴파일 (학습하지 않음)
        # final_model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        #     loss="sparse_categorical_crossentropy",
        #     metrics=["accuracy"]
        # )
        # print("Final model compiled successfully with averaged weights.")
        # # 콜백 정의
        # early_stopping_final = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        # lr_scheduler_final = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)
        
        # # 최종 모델을 전체 train 데이터로 추가 학습
        # full_train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(8).shuffle(1000)
        # history_final = final_model.fit(
        #     full_train_dataset,
        #     epochs=50,
        #     callbacks=[lr_scheduler_final]
        # )

        # # 최종 모델 가중치 저장
        # final_model_weights_path = os.path.join(subject_save_path, "final_model.weights.h5")
        # final_model.save_weights(final_model_weights_path)
        # print(f"Final model weights saved for subject {subject} at {final_model_weights_path}.")

    #    # 테스트 데이터셋 생성
    #     test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)

    #     # 앙상블 예측 수행
    #     ensemble_predicted_labels = ensemble_predictions(fold_model_paths, test_dataset)

        # 테스트 데이터셋 결과 레포트
        test_report = classification_report(
            test_labels,
            ensemble_predicted_labels,
            target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"],
            labels=[0, 1, 2, 3, 4],
            zero_division=0
        )
        print(f"\nTest Report for {subject} (Ensemble):\n{test_report}")

        # 혼동 행렬 출력
        cm = confusion_matrix(test_labels, ensemble_predicted_labels)
        print(f"Confusion Matrix for {subject}:\n{cm}")

        # ROC AUC Score 계산 (멀티 클래스)
        try:
            roc_auc = roc_auc_score(
                tf.keras.utils.to_categorical(test_labels),
                tf.keras.utils.to_categorical(ensemble_predicted_labels),
                multi_class='ovr'
            )
            print(f"ROC AUC Score for {subject}: {roc_auc}")
        except Exception as e:
            print(f"ROC AUC Score could not be calculated for {subject}: {e}")

        # 테스트 레포트 저장
        final_model_dir = os.path.join(SAVE_PATH, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        with open(os.path.join(final_model_dir, f"{subject}_test_report.txt"), "w") as f:
            f.write(test_report)

    print("\n=== Training Completed ===")


# 실행
train_subjects_with_kfold()
