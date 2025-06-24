#boosting 대신 여러 모델의 가중치를 평균내어 최종 모델을 만드는 방식 사용
#여기서 ROC AUC 오류가 나오는 이유는 감정 라벨링을 하지 않았기 때문
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.layers import Conv3D, Dense, LayerNormalization, Dropout, GlobalAveragePooling3D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb  # 부스팅 메타모델용

# 메모리 제한
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
DATA_PATH = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_de_features_2D_mapping"
SAVE_PATH = "/home/bcml1/sigenv/_1주차_eeg_1s_inter_DEAP/result2"
os.makedirs(SAVE_PATH, exist_ok=True)

# 감정 라벨 매핑
EMOTION_MAPPING = {
    "Excited": 0,
    "Relaxed": 1,
    "Stressed": 2,
    "Bored": 3,
    "Neutral": 4
}

# 디버그 함수 (필요시 사용)
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

# 데이터 전처리 함수
def preprocess_data(data):
    # 여기서는 NaN/Inf 처리만 진행 (필요 시 정규화 활성화)
    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
    data = tf.where(tf.math.is_inf(data), tf.zeros_like(data), data)
    # min_val = tf.reduce_min(data)
    # max_val = tf.reduce_max(data)
    # range_val = max_val - min_val + 1e-8  # 분모가 0이 되지 않도록 작은 값을 추가
    # data = (data - min_val) / range_val
    return data

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
    
# 데이터 로드 함수 (파일명: "sXX_sample_XX_segment_XXX_label_[Emotion]_2D.npy")
def load_subject_data(subject, data_path=DATA_PATH):
    data, labels = [], []
    for file_name in os.listdir(data_path):
        if file_name.startswith(subject) and file_name.endswith("_2D.npy"):
            parts = file_name.split("_")
            if len(parts) < 7:
                print(f"Unexpected file format: {file_name}")
                continue
            # parts[4] 예: "segment_003"
            segment_name = parts[4]
            if segment_name in ["000", "001", "002"]:
                print(f"Skipping segment: {segment_name}")
                continue
            # 감정 라벨은 파일 이름의 마지막에서 두 번째 항목
            emotion_label = parts[-2]
            if emotion_label in EMOTION_MAPPING:
                label = EMOTION_MAPPING[emotion_label]
                file_path = os.path.join(data_path, file_name)
                try:
                    de_features = np.load(file_path)  # (4, 6, 6)
                    de_features = np.expand_dims(de_features, axis=-1)  # (4, 6, 6, 1)
                    data.append(de_features)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
            else:
                print(f"Unknown emotion label: {emotion_label} in file {file_name}")
    data_array = np.array(data)
    labels_array = np.array(labels)
    if data_array.size == 0:
        print(f"Warning: No data found for {subject}")
    else:
        print(f"Loaded data shape for {subject}: {data_array.shape}")
    return data_array, labels_array

# 메타모델(부스팅) 및 스태킹 앙상블 함수
# def ensemble_predictions_with_boosting(model_paths, test_dataset, input_shape, meta_model):
#     # 기본 모델들의 예측을 수집
#     base_predictions = []
#     for path in model_paths:
#         model = TransformerEncoder(input_shape=input_shape, n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
#         model.build(input_shape=(None,) + input_shape)  # (batch, 4,6,6,1)
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#             loss="sparse_categorical_crossentropy",
#             metrics=["accuracy"]
#         )
#         model.load_weights(path)
#         preds = model.predict(test_dataset, verbose=0)  # (num_samples, 5)
#         base_predictions.append(preds)
#     base_predictions = np.array(base_predictions)  # (num_models, num_samples, 5)
#     base_predictions_mean = np.mean(base_predictions, axis=0)  # (num_samples, 5)
#     # 메타모델을 사용하여 최종 예측 (XGBoost 메타모델)
#     final_preds_proba = meta_model.predict_proba(base_predictions_mean)
#     final_preds = np.argmax(final_preds_proba, axis=1)
#     return final_preds

# -------------------------
# Inter-Subject CV (Weight Averaging) 코드
# -------------------------

# Inter-Subject Cross Validation 함수 (train_inter_subject_cv)
def train_inter_subject_cv():
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 33)]  # 예: s01 ~ s32
    # 여기서는 5-Fold CV (피실험자 단위) 적용
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    subjects = np.array(subjects)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        print(f"\n=== Inter-Subject CV Fold {fold+1} ===")
        train_subjects = subjects[train_idx]
        test_subjects = subjects[test_idx]
        print(f"Train subjects: {train_subjects}")
        print(f"Test subjects: {test_subjects}")
        
        # 훈련 피실험자의 데이터 로드 및 결합
        train_data_list, train_labels_list = [], []
        for subj in train_subjects:
            data, labels = load_subject_data(subj, DATA_PATH)
            if data.size > 0:
                train_data_list.append(data)
                train_labels_list.append(labels)
        if len(train_data_list) == 0:
            print("No training data found. Skipping this fold.")
            continue
        train_data = np.concatenate(train_data_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        # 라벨 필터링 적용
        valid_labels = [0, 1, 2, 3, 4]
        valid_indices = np.isin(train_labels, valid_labels)
        train_data = train_data[valid_indices]
        train_labels = train_labels[valid_indices].astype(np.int32)
        
        # 테스트 피실험자의 데이터 로드 및 결합
        test_data_list, test_labels_list = [], []
        for subj in test_subjects:
            data, labels = load_subject_data(subj, DATA_PATH)
            if data.size > 0:
                test_data_list.append(data)
                test_labels_list.append(labels)
        if len(test_data_list) == 0:
            print("No testing data found. Skipping this fold.")
            continue
        test_data = np.concatenate(test_data_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)
        valid_indices = np.isin(test_labels, valid_labels)
        test_data = test_data[valid_indices]
        test_labels = test_labels[valid_indices].astype(np.int32)
        
        # 데이터셋 생성
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(8).shuffle(1000)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
        
        # 한 폴드에서 n_models 개의 기본 모델 학습 후 가중치 저장
        n_models = 3  # 예시로 3개의 모델을 학습합니다.
        weight_paths = []
        for i in range(n_models):
            print(f"Training base model {i+1}/{n_models} for Fold {fold+1}...")
            base_model = TransformerEncoder(input_dim=(4, 6, 6, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
            base_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            base_model.fit(
                train_dataset,
                epochs=5,
                validation_data=test_dataset,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
                ]
            )
            model_path = os.path.join(SAVE_PATH, f"fold_{fold+1}_model_{i+1}.weights.h5")
            base_model.save_weights(model_path)
            weight_paths.append(model_path)
            # 메모리 정리
            tf.keras.backend.clear_session()
        
        # 평균 가중치를 적용할 새로운 모델 생성 후, 저장된 가중치들을 평균합니다.
        print(f"Averaging weights for Fold {fold+1} using models: {weight_paths}")
        averaged_model = TransformerEncoder(input_dim=(4, 6, 6, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
        averaged_model.build((None, 4, 6, 6, 1))
        averaged_model = average_model_weights(averaged_model, weight_paths)
        
        # 테스트 데이터에 대해 예측 수행
        preds = averaged_model.predict(test_dataset, verbose=0)
        final_preds = np.argmax(preds, axis=1)
        
        print(f"\nTest Report for Fold {fold+1}")
        report = classification_report(
            test_labels,
            final_preds,
            target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"],
            labels=[0, 1, 2, 3, 4],
            zero_division=0
        )
        print(report)
        cm = confusion_matrix(test_labels, final_preds)
        print("Confusion Matrix:")
        print(cm)
        try:
            roc_auc = roc_auc_score(
                tf.keras.utils.to_categorical(test_labels),
                tf.keras.utils.to_categorical(final_preds),
                multi_class='ovr'
            )
            print(f"ROC AUC Score: {roc_auc}")
        except Exception as e:
            print(f"ROC AUC Score could not be calculated: {e}")
        
        # 테스트 레포트 저장
        fold_report_path = os.path.join(SAVE_PATH, "final_model", f"fold_{fold+1}_test_report.txt")
        os.makedirs(os.path.dirname(fold_report_path), exist_ok=True)
        with open(fold_report_path, "w") as f:
            f.write(report)
        
        # 메모리 정리
        tf.keras.backend.clear_session()

    print("\n=== Inter-Subject Training Completed ===")


# 실행
train_inter_subject_cv()
