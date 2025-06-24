import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.layers import Conv3D, Dense, LayerNormalization, Dropout, GlobalAveragePooling3D, Flatten
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

# 디버그 함수 (필요 시 사용)
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

# 데이터 전처리 함수 (필요 시 추가 전처리 가능)
def preprocess_data(data):
    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
    data = tf.where(tf.math.is_inf(data), tf.zeros_like(data), data)
    return data

# Spatial-Spectral Convolution Module (변경 없음)
class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.spatial_conv = Conv3D(filters, kernel_size=(1, 3, 3), strides=strides,
                                   padding="same", activation="relu")
        self.spectral_conv = Conv3D(filters, kernel_size=(4, 1, 1), strides=strides,
                                    padding="same", activation="relu")
    def call(self, inputs):
        spatial_features = self.spatial_conv(inputs)
        spectral_features = self.spectral_conv(inputs)
        return spatial_features + spectral_features
    def get_config(self):
        return {
            "filters": self.spatial_conv.filters,
            "kernel_size": self.spatial_conv.kernel_size,
            "strides": self.spatial_conv.strides,
        }

# Spatial and Spectral Attention Branch (디버그 출력 제거)
class SpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialSpectralAttention, self).__init__()
        self.spatial_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")
        self.spectral_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")
    def call(self, inputs):
        spatial_mask = self.spatial_squeeze(inputs)
        spatial_output = inputs * spatial_mask
        spectral_mask = self.spectral_squeeze(inputs)
        spectral_output = inputs * spectral_mask
        combined_output = spatial_output + spectral_output
        # 디버그 출력 제거
        return combined_output
    def get_config(self):
        return {}

# Transformer Encoder Layer (변경 없음)
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
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
    def get_config(self):
        return {
            "d_model": self.mha.key_dim * self.mha.num_heads,
            "n_heads": self.mha.num_heads,
            "d_ff": self.ffn.layers[0].units,
            "dropout_rate": self.dropout1.rate,
        }

# Transformer Encoder 모델 (입력 인자는 input_shape 사용)
class TransformerEncoder(tf.keras.Model):
    def __init__(self, input_shape, n_layers=6, n_heads=8, d_ff=2048, p_drop=0.3, d_model=64):
        super(TransformerEncoder, self).__init__()
        self.input_shape_ = input_shape  # 전체 입력 형태 저장
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.d_model = d_model

        self.attention = SpatialSpectralAttention()
        self.conv_block1 = SpatialSpectralConvModule(8, kernel_size=(1, 3, 3), strides=(1, 3, 3))
        self.conv_block2 = SpatialSpectralConvModule(16, kernel_size=(4, 1, 1), strides=(4, 1, 1))
        self.conv_block3 = SpatialSpectralConvModule(32, kernel_size=(1, 2, 2), strides=(1, 2, 2))
        self.flatten = Flatten()
        self.dense_projection = Dense(d_model, activation="relu")
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)
            for _ in range(n_layers)
        ]
        self.output_dense = Dense(5, activation="softmax")
    def build(self, input_shape):
        super(TransformerEncoder, self).build(input_shape)
    def call(self, inputs, training=False):
        x = preprocess_data(inputs)
        x = self.attention(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense_projection(x)
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        # squeeze()로 불필요한 차원 제거 (출력 shape: (batch_size, 5))
        return tf.squeeze(self.output_dense(x), axis=1)
    def get_config(self):
        return {
            "input_shape": self.input_shape_,
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
            segment_name = parts[4]
            if segment_name in ["000", "001", "002"]:
                print(f"Skipping segment: {segment_name}")
                continue
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

# 메타모델(부스팅) 및 스태킹 앙상블 함수 (XGBoost 메타모델 사용)
def ensemble_predictions_with_boosting(model_paths, test_dataset, input_shape, meta_model):
    base_predictions = []
    for path in model_paths:
        model = TransformerEncoder(input_shape=input_shape, n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
        model.build(input_shape=(None,) + input_shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.load_weights(path)
        preds = model.predict(test_dataset, verbose=0)
        base_predictions.append(preds)
    base_predictions = np.array(base_predictions)
    base_predictions_mean = np.mean(base_predictions, axis=0)
    final_preds_proba = meta_model.predict_proba(base_predictions_mean)
    final_preds = np.argmax(final_preds_proba, axis=1)
    return final_preds

# Intra-Subject Cross Validation 구현 (한 피실험자 내에서 K-Fold 교차 검증)
def train_intra_subject_cv():
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 33)]  # 예: s01 ~ s32
    for subject in subjects:
        print(f"\n=== Training Intra-Subject CV for {subject} ===")
        # 각 피실험자별 데이터 로드
        subject_data, subject_labels = load_subject_data(subject, DATA_PATH)
        if subject_data.size == 0:
            print(f"No data found for {subject}. Skipping...")
            continue

        subject_data = np.nan_to_num(subject_data, nan=0.0, posinf=0.0, neginf=0.0)
        # 피실험자 데이터는 모두 동일한 subject에서 수집되었으므로, intra-subject CV를 위해 KFold 적용
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold_model_paths = []
        val_predictions_list = []
        val_labels_list = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(subject_data)):
            print(f"\n--- Fold {fold+1} for subject {subject} ---")
            train_data = subject_data[train_idx]
            train_labels = subject_labels[train_idx]
            val_data = subject_data[val_idx]
            val_labels = subject_labels[val_idx]
            
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(8).shuffle(1000)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(8)
            
            model = TransformerEncoder(input_shape=(4, 6, 6, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
            ])
            
            fold_model_path = os.path.join(SAVE_PATH, subject, f"{subject}_fold_{fold+1}.weights.h5")
            os.makedirs(os.path.dirname(fold_model_path), exist_ok=True)
            model.save_weights(fold_model_path)
            fold_model_paths.append(fold_model_path)
            
            # 검증 데이터 예측 저장 (메타모델 학습용)
            val_preds = model.predict(val_dataset, verbose=0)  # (num_val_samples, 5)
            val_predictions_list.append(val_preds)
            val_labels_list.append(val_labels)
            
            del model
            tf.keras.backend.clear_session()
        
        # 메타모델 학습 (XGBoost 사용)
        val_predictions = np.concatenate(val_predictions_list, axis=0)  # (total_val_samples, 5)
        val_labels = np.concatenate(val_labels_list, axis=0)  # (total_val_samples,)
        meta_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=5,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        meta_model.fit(val_predictions, val_labels)
        print(f"Meta-model trained for subject {subject}.")
        
        # 전체 데이터를 대상으로 기본 모델 예측 수집
        base_preds = []
        for path in fold_model_paths:
            model = TransformerEncoder(input_shape=(4, 6, 6, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
            model.build(input_shape=(None,) + (4, 6, 6, 1))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            model.load_weights(path)
            preds = model.predict(subject_data, verbose=0)  # (num_samples, 5)
            base_preds.append(preds)
            del model
            tf.keras.backend.clear_session()
        base_preds = np.array(base_preds)
        base_preds_mean = np.mean(base_preds, axis=0)
        
        # 메타모델로 최종 예측
        final_preds_proba = meta_model.predict_proba(base_preds_mean)
        final_preds = np.argmax(final_preds_proba, axis=1)
        
        # 평가
        report = classification_report(subject_labels, final_preds, target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"], labels=[0,1,2,3,4], zero_division=0)
        print(f"\nTest Report for subject {subject} (Intra-Subject CV with Boosting):\n{report}")
        cm = confusion_matrix(subject_labels, final_preds)
        print("Confusion Matrix:")
        print(cm)
        try:
            roc_auc = roc_auc_score(
                tf.keras.utils.to_categorical(subject_labels),
                tf.keras.utils.to_categorical(final_preds),
                multi_class='ovr'
            )
            print(f"ROC AUC Score: {roc_auc}")
        except Exception as e:
            print(f"ROC AUC Score could not be calculated: {e}")
        
        # 결과 저장
        report_path = os.path.join(SAVE_PATH, "final_model", f"{subject}_test_report.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
            
    print("\n=== Intra-Subject Training Completed ===")

# 실행
train_intra_subject_cv()
