import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.layers import Conv3D, Dense, LayerNormalization, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# 데이터 및 결과 경로 설정
DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
SAVE_PATH = "/home/bcml1/sigenv/_1주차_eeg_1s_inter_DEAP/norm_result3"
os.makedirs(SAVE_PATH, exist_ok=True)

# 감정 라벨 매핑
EMOTION_MAPPING = {
    "Excited": 0,
    "Relaxed": 1,
    "Stressed": 2,
    "Bored": 3,
    "Neutral": 4
}

# ---------------------------
# 공통 함수 및 모델 클래스들
# ---------------------------

# 데이터 전처리 함수: NaN, Inf를 0으로 치환
def preprocess_data(data):
    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)  # NaN 값을 0으로 대체
    data = tf.where(tf.math.is_inf(data), tf.zeros_like(data), data)  # Inf 값을 0으로 대체
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)
    range_val = max_val - min_val + 1e-8  # 분모가 0이 되지 않도록 작은 값을 추가
    data = (data - min_val) / range_val
    return data

# 여러 모델의 가중치를 평균내어 최종 모델에 적용하는 함수
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
            # 올바른 입력 shape: (None,) + input_dim
            temp_model.build((None,) + model.input_dim)
            temp_model.load_weights(path)
            weights.append(temp_model.get_weights())
        except Exception as e:
            print(f"Error loading weights from {path}: {e}")
            continue

    if len(weights) == 0:
        raise ValueError("No valid models were loaded. Cannot average weights.")

    # 각 weight 배열별로 평균내기
    avg_weights = [np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))]
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
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)
    
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
        # 인스턴스 변수 (average_model_weights()에서 활용)
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.d_model = d_model

        # 모델 구성 요소
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
        super(TransformerEncoder, self).build(input_shape)

    def call(self, inputs, training=False):
        x = preprocess_data(inputs)
        x = self.attention(x)  # Spatial-Spectral Attention 적용
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense_projection(x)
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        # (batch, 1, 5) 형태로 나온 후 squeeze
        return tf.squeeze(self.output_dense(x), axis=1)

    def get_config(self):
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

# 피실험자별 데이터 로드 함수
# 파일명 예: "sXX_sample_XX_segment_XXX_label_[Emotion]_2D_DE.npy"
def load_subject_data(subject, data_path=DATA_PATH):
    data, labels = [], []
    for file_name in os.listdir(DATA_PATH):
        if file_name.startswith(subject) and file_name.endswith("_2D_DE.npy"):
            # Segment 이름 추출
            segment_name = file_name.split("_")[4]  # e.g., "segment_00"
            # print(f"check segment: {segment_name}")
            if segment_name in ["000", "001", "002"]:
                print(f"skipping segment: {segment_name}")
                continue  # Skip these segments
            
            emotion_label = file_name.split("_")[-3]
            #print(f"Emotion labels:{emotion_label}")
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

# --------------------------------------------
# 10‑fold CV로 모델 학습 후 최종 모델 생성 및 평가
# --------------------------------------------
def train_inter_subject_cv_final_model():
    # 전체 피실험자: s01 ~ s32
    all_subjects = np.array([f"s{str(i).zfill(2)}" for i in range(1, 33)])
    
    # 전체 피실험자 섞은 후, 20%는 최종 테스트셋, 나머지는 최종 학습셋으로 사용
    np.random.seed(42)
    np.random.shuffle(all_subjects)
    n_test = int(len(all_subjects) * 0.2)
    final_test_subjects = all_subjects[:n_test]
    final_train_subjects = all_subjects[n_test:]
    
    print("Final Train Subjects:", final_train_subjects)
    print("Final Test Subjects:", final_test_subjects)
    
    # 10-fold CV: final_train_subjects를 대상으로 fold마다 하나의 모델 학습
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_weight_paths = []
    fold_idx = 1
    for train_idx, val_idx in kf.split(final_train_subjects):
        print(f"\n=== CV Fold {fold_idx} ===")
        fold_train_subjects = final_train_subjects[train_idx]
        fold_val_subjects = final_train_subjects[val_idx]
        
        # fold의 train 데이터 로드
        fold_train_data_list, fold_train_labels_list = [], []
        for subj in fold_train_subjects:
            data, labels = load_subject_data(subj, DATA_PATH)
            if data.size > 0:
                fold_train_data_list.append(data)
                fold_train_labels_list.append(labels)
        if len(fold_train_data_list) == 0:
            print(f"No training data in fold {fold_idx}")
            fold_idx += 1
            continue
        fold_train_data = np.concatenate(fold_train_data_list, axis=0)
        fold_train_labels = np.concatenate(fold_train_labels_list, axis=0)
        valid_idx = np.isin(fold_train_labels, [0,1,2,3,4])
        fold_train_data = fold_train_data[valid_idx]
        fold_train_labels = fold_train_labels[valid_idx].astype(np.int32)
        
        # fold의 validation 데이터 로드
        fold_val_data_list, fold_val_labels_list = [], []
        for subj in fold_val_subjects:
            data, labels = load_subject_data(subj, DATA_PATH)
            if data.size > 0:
                fold_val_data_list.append(data)
                fold_val_labels_list.append(labels)
        if len(fold_val_data_list) == 0:
            print(f"No validation data in fold {fold_idx}")
            fold_idx += 1
            continue
        fold_val_data = np.concatenate(fold_val_data_list, axis=0)
        fold_val_labels = np.concatenate(fold_val_labels_list, axis=0)
        valid_idx = np.isin(fold_val_labels, [0,1,2,3,4])
        fold_val_data = fold_val_data[valid_idx]
        fold_val_labels = fold_val_labels[valid_idx].astype(np.int32)
        
        # TensorFlow 데이터셋 생성
        fold_train_dataset = tf.data.Dataset.from_tensor_slices((fold_train_data, fold_train_labels)).batch(8).shuffle(1000)
        fold_val_dataset = tf.data.Dataset.from_tensor_slices((fold_val_data, fold_val_labels)).batch(32)
        
        # 각 fold에서 모델 학습 (Epoch 수는 필요에 따라 조정)
        model = TransformerEncoder(input_dim=(4, 6, 6, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        # model.fit(fold_train_dataset, epochs=50, validation_data=fold_val_dataset, callbacks=[
        #     EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        #     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        # ])
        model.fit(fold_train_dataset, epochs=50, validation_data=fold_val_dataset, callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        ])
        # fold 모델 가중치 저장
        weight_path = os.path.join(SAVE_PATH, f"cv_fold_{fold_idx}.weights.h5")
        model.save_weights(weight_path)
        fold_weight_paths.append(weight_path)
        
        # ------ 각 fold의 validation 결과 평가 후 txt 파일로 저장 -------
        fold_val_preds = model.predict(fold_val_dataset, verbose=0)
        fold_val_preds = np.argmax(fold_val_preds, axis=1)
        fold_report = classification_report(
            fold_val_labels,
            fold_val_preds,
            target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"],
            labels=[0, 1, 2, 3, 4],
            zero_division=0
        )
        fold_cm = confusion_matrix(fold_val_labels, fold_val_preds)
        fold_output_path = os.path.join(SAVE_PATH, f"fold_{fold_idx}_report.txt")
        with open(fold_output_path, "w") as f:
            f.write("Classification Report:\n")
            f.write(fold_report + "\n\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(fold_cm) + "\n")
        print(f"Fold {fold_idx} report saved to: {fold_output_path}")
        # ----------------------------------------------------------
        
        fold_idx += 1
        tf.keras.backend.clear_session()
    
    # 10-fold CV에서 저장된 모든 모델의 가중치를 평균내어 최종 모델 생성
    print("\nAveraging weights from CV models...")
    final_model = TransformerEncoder(input_dim=(4, 6, 6, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
    final_model.build((None, 4, 6, 6, 1))
    final_model = average_model_weights(final_model, fold_weight_paths)
    
    # 최종 모델 Fine-tuning을 위해, final_train_subjects 전체 데이터를 로드
    all_train_data_list, all_train_labels_list = [], []
    for subj in final_train_subjects:
        data, labels = load_subject_data(subj, DATA_PATH)
        if data.size > 0:
            all_train_data_list.append(data)
            all_train_labels_list.append(labels)
    if len(all_train_data_list) > 0:
        all_train_data = np.concatenate(all_train_data_list, axis=0)
        all_train_labels = np.concatenate(all_train_labels_list, axis=0)
        valid_idx = np.isin(all_train_labels, [0,1,2,3,4])
        all_train_data = all_train_data[valid_idx]
        all_train_labels = all_train_labels[valid_idx].astype(np.int32)
        train_dataset = tf.data.Dataset.from_tensor_slices((all_train_data, all_train_labels)).batch(8).shuffle(1000)
    else:
        print("No training data available for final fine-tuning.")
        return
    
    # Final model을 전체 train 데이터로 fine-tuning
    print("\nFine-tuning final model on all training data...")
    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
    # final_model.fit(train_dataset, epochs=50, callbacks=[
    #     EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
    #     ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)
    # ])
    final_model.fit(train_dataset, epochs=50, callbacks=[
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)
    ])
    
    # 최종 테스트셋(final test subjects) 데이터 로드
    all_test_data_list, all_test_labels_list = [], []
    for subj in final_test_subjects:
        data, labels = load_subject_data(subj, DATA_PATH)
        if data.size > 0:
            all_test_data_list.append(data)
            all_test_labels_list.append(labels)
    if len(all_test_data_list) > 0:
        all_test_data = np.concatenate(all_test_data_list, axis=0)
        all_test_labels = np.concatenate(all_test_labels_list, axis=0)
        valid_idx = np.isin(all_test_labels, [0,1,2,3,4])
        all_test_data = all_test_data[valid_idx]
        all_test_labels = all_test_labels[valid_idx].astype(np.int32)
        test_dataset = tf.data.Dataset.from_tensor_slices((all_test_data, all_test_labels)).batch(32)
    else:
        print("No test data available.")
        return
    
    # 최종 모델을 최종 테스트 데이터로 평가
    print("\nEvaluating final model on final test data...")
    preds = final_model.predict(test_dataset, verbose=0)
    final_preds = np.argmax(preds, axis=1)
    report = classification_report(
        all_test_labels,
        final_preds,
        target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"],
        labels=[0,1,2,3,4],
        zero_division=0
    )
    print(report)
    cm = confusion_matrix(all_test_labels, final_preds)
    print("Confusion Matrix:")
    print(cm)
    try:
        roc_auc = roc_auc_score(
            tf.keras.utils.to_categorical(all_test_labels),
            tf.keras.utils.to_categorical(final_preds),
            multi_class='ovr'
        )
        print(f"ROC AUC Score: {roc_auc}")
    except Exception as e:
        print(f"ROC AUC Score could not be calculated: {e}")
    
    # 최종 모델 가중치 저장
    final_model_path = os.path.join(SAVE_PATH, "final_model.weights.h5")
    final_model.save_weights(final_model_path)
    print(f"Final model weights saved to: {final_model_path}")

    # 결과를 txt 파일로 저장하는 예시 코드 추가
    output_path = os.path.join(SAVE_PATH, "final_test_report.txt")
    with open(output_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n\n")
        try:
            f.write(f"ROC AUC Score: {roc_auc}\n")
        except Exception:
            f.write("ROC AUC Score could not be calculated.\n")
            
    print(f"Final test report saved to: {output_path}")
    
    
# 최종 프로세스 실행
train_inter_subject_cv_final_model()
