#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEAP 데이터셋 (EEG, PPG) 기반 멀티모달 감정 분류 모델
- 각 피험자(s01 ~ s22)는 이미 전처리되어 저장된 데이터 파일을 사용합니다.
- EEG 데이터: (40, 32, 128) → 모델 입력: (40, 32, 128, 1)
- PPG 데이터: (40, 128) → 모델 입력: (40, 128, 1, 1)
- 라벨: (40,) (Negative: 0, Positive: 1, Neutral: 2)
- 각 파일에는 40개의 1초 샘플이 저장되어 있으며, 이들 샘플은 라벨 파일과 1:1 매칭됩니다.
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, Dropout, AveragePooling2D, DepthwiseConv2D,
    LayerNormalization, MultiHeadAttention, Input, Lambda, Add, Concatenate, Softmax
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =============================================================================
# GPU 메모리 제한 (필요 시)
# =============================================================================
def limit_gpu_memory(memory_limit_mib=8000):
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

limit_gpu_memory(8000)

# =============================================================================
# 하이퍼파라미터 및 경로 설정
# =============================================================================
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4

# 데이터 경로 (필요에 따라 수정)
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG"
PPG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_PPG"
LABELS_PATH  = "/home/bcml1/2025_EMOTION/DEAP_three_labels"
SAVE_PATH    = "/home/bcml1/sigenv/_3주차_eeg+pps_CNN/result2"
os.makedirs(SAVE_PATH, exist_ok=True)

# =============================================================================
# 1. Dual-Stream Feature Extractor (EEG & PPG)
# =============================================================================
def create_dual_stream_feature_extractor():
    """
    EEG와 PPG를 각각 CNN으로 특징 추출
    - EEG Branch: 입력 (32, 128, 1)
    - PPG Branch: 입력 (128, 1, 1)
    """
    ## EEG Branch
    eeg_input = Input(shape=(32, 128, 1), name="EEG_Input")
    x = Conv2D(filters=16, kernel_size=(4,16), strides=(2,8), padding='valid', activation='relu')(eeg_input)
    x = DepthwiseConv2D(kernel_size=(6,6), strides=(3,3), padding='valid', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)  # 추가 dropout
    eeg_output = Dense(128, activation="relu")(x)
    
    ## PPG Branch
    ppg_input = Input(shape=(128, 1, 1), name="PPG_Input")
    y = Conv2D(filters=16, kernel_size=(4,1), strides=(2,1), padding='valid', activation='relu')(ppg_input)
    y = DepthwiseConv2D(kernel_size=(6,1), strides=(3,1), padding='valid', activation='relu')(y)
    y = AveragePooling2D(pool_size=(2,1), strides=(2,1))(y)
    y = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), activation='relu')(y)
    y = Flatten()(y)
    y = Dropout(0.5)(y)  # 추가 dropout
    ppg_output = Dense(128, activation="relu")(y)
    
    model = Model(inputs=[eeg_input, ppg_input], outputs=[eeg_output, ppg_output],
                  name="DualStreamFeatureExtractor")
    return model

# =============================================================================
# 2. Inter-Modality Fusion Module (Cross-Modal Transformer)
# =============================================================================
def create_inter_modality_fusion(eeg_features, ppg_features, num_heads=4, d_model=128, dropout_rate=0.1):
    """
    EEG와 PPG 간의 상호관계를 학습하는 Cross‑Modal Transformer
    """
    # EEG → PPG
    eeg_query = Dense(d_model)(eeg_features)
    ppg_key_value = Dense(d_model)(ppg_features)
    eeg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_query)  # (batch,1,d_model)
    ppg_key_value = Lambda(lambda x: tf.expand_dims(x, axis=1))(ppg_key_value)
    
    cross_att_1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_EEG_to_PPG")(
        query=eeg_query, key=ppg_key_value, value=ppg_key_value)
    cross_att_1 = Dropout(dropout_rate)(cross_att_1)
    cross_att_1 = Add()([eeg_query, cross_att_1])
    cross_att_1 = LayerNormalization(epsilon=1e-6)(cross_att_1)
    cross_att_1 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_att_1)  # (batch,d_model)
    
    # PPG → EEG
    ppg_query = Dense(d_model)(ppg_features)
    ppg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(ppg_query)
    eeg_key_value_2 = Dense(d_model)(eeg_features)
    eeg_key_value_2 = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_key_value_2)
    
    cross_att_2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_PPG_to_EEG")(
        query=ppg_query, key=eeg_key_value_2, value=eeg_key_value_2)
    cross_att_2 = Dropout(dropout_rate)(cross_att_2)
    cross_att_2 = Add()([ppg_query, cross_att_2])
    cross_att_2 = LayerNormalization(epsilon=1e-6)(cross_att_2)
    cross_att_2 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_att_2)
    
    fused_features = Concatenate(axis=-1)([cross_att_1, cross_att_2])
    fused_features = Dense(d_model, activation="relu", name="Fused_Linear")(fused_features)
    
    # Self-Attention Fusion
    fused_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(fused_features)
    self_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttentionFusion")(
        query=fused_expanded, key=fused_expanded, value=fused_expanded)
    self_att = Dropout(dropout_rate)(self_att)
    self_att = Add()([fused_expanded, self_att])
    self_att = LayerNormalization(epsilon=1e-6)(self_att)
    self_att = Lambda(lambda x: tf.squeeze(x, axis=1))(self_att)
    return self_att

# =============================================================================
# 3. Intra-Modality Encoding Module
# =============================================================================
def create_intra_modality_encoding(eeg_features, ppg_features, num_heads=4, d_model=128, dropout_rate=0.1):
    """
    EEG와 PPG 각각의 고유 특성을 보존 및 강화하는 Intra‑Modal Encoding
    """
    # EEG Self-Attention
    eeg_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_features)
    eeg_self_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_EEG")(
        query=eeg_expanded, key=eeg_expanded, value=eeg_expanded)
    eeg_self_att = Dropout(dropout_rate)(eeg_self_att)
    eeg_self_att = Add()([eeg_expanded, eeg_self_att])
    eeg_self_att = LayerNormalization(epsilon=1e-6)(eeg_self_att)
    eeg_self_att = Lambda(lambda x: tf.squeeze(x, axis=1))(eeg_self_att)
    
    # PPG Self-Attention
    ppg_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(ppg_features)
    ppg_self_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_PPG")(
        query=ppg_expanded, key=ppg_expanded, value=ppg_expanded)
    ppg_self_att = Dropout(dropout_rate)(ppg_self_att)
    ppg_self_att = Add()([ppg_expanded, ppg_self_att])
    ppg_self_att = LayerNormalization(epsilon=1e-6)(ppg_self_att)
    ppg_self_att = Lambda(lambda x: tf.squeeze(x, axis=1))(ppg_self_att)
    
    return eeg_self_att, ppg_self_att

# =============================================================================
# 4. 전체 모델 구성 (EEG, PPG 입력 → Dual‑Stream Extraction → Fusion → 분류)
# =============================================================================
def build_combined_model(num_classes=3):
    """
    EEG와 PPG 데이터를 받아서  
      - Dual‑Stream Feature Extraction  
      - Cross‑Modal Fusion  
      - Intra‑Modal Encoding  
    을 거쳐 최종 분류 결과를 출력하는 모델 구성
    """
    # 한 피험자의 40개 샘플 데이터 입력
    eeg_input = Input(shape=(32, 128, 1), name="EEG_Input")
    ppg_input = Input(shape=(128, 1, 1), name="PPG_Input")
    
    dual_extractor = create_dual_stream_feature_extractor()
    eeg_features, ppg_features = dual_extractor([eeg_input, ppg_input])
    
    fused_features = create_inter_modality_fusion(eeg_features, ppg_features)
    eeg_encoded, ppg_encoded = create_intra_modality_encoding(eeg_features, ppg_features)
    
    inter_classification = Dense(num_classes, activation="softmax", name="Inter_Classification")(fused_features)
    eeg_classification   = Dense(num_classes, activation="softmax", name="EEG_Classification")(eeg_encoded)
    ppg_classification   = Dense(num_classes, activation="softmax", name="PPG_Classification")(ppg_encoded)
    
    concat_features = Concatenate()([fused_features, eeg_encoded, ppg_encoded])
    weights_logits = Dense(units=3, activation=None, name="Weight_Logits")(concat_features)
    weights = Softmax(axis=-1, name="Weight_Softmax")(weights_logits)
    
    model = Model(inputs=[eeg_input, ppg_input],
                  outputs=[inter_classification, eeg_classification, ppg_classification, weights],
                  name="Multimodal_Emotion_Classifier")
    return model

# =============================================================================
# 5. Custom 학습 단계 포함 모델 클래스 정의
# =============================================================================
class MultimodalEmotionClassifier(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super(MultimodalEmotionClassifier, self).__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        # y가 dict이면 Inter_Classification 키 사용
        if isinstance(y, dict):
            y_true = y["Inter_Classification"]
        else:
            y_true = y

        with tf.GradientTape() as tape:
            # 네 개의 출력을 모두 받음
            inter_pred, eeg_pred, ppg_pred, weights = self(x, training=True)
            # 세 branch의 분류 loss 계산
            loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
            loss_eeg   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eeg_pred)
            loss_ppg   = tf.keras.losses.sparse_categorical_crossentropy(y_true, ppg_pred)
            # 동적 가중치 weights는 각 loss에 대한 가중치를 의미하므로, 이 weights로 가중합을 구성
            # weights shape: (batch, 3)이고, 각 loss는 (batch,) shape이므로, loss들을 스택하여 가중합
            losses = tf.stack([loss_inter, loss_eeg, loss_ppg], axis=1)  # shape: (batch, 3)
            weighted_loss = tf.reduce_sum(weights * losses, axis=1)
            loss = tf.reduce_mean(weighted_loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y_true, inter_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y = data
        y_true = y["Inter_Classification"] if isinstance(y, dict) else y
        inter_pred, _, _, _ = self(x, training=False)
        loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
        loss = tf.reduce_mean(loss_inter)
        self.compiled_metrics.update_state(y_true, inter_pred)
        metric_results = {m.name: m.result() for m in self.metrics}
        results = {"accuracy": metric_results.get("accuracy", 0.0), "loss": loss}
        return results

# =============================================================================
# 6. 클래스 빈도 기반 alpha 및 Focal Loss
# =============================================================================
def compute_class_weights(labels, num_classes=3):
    class_counts = np.bincount(labels, minlength=num_classes)
    total = np.sum(class_counts)
    class_weights = total / (num_classes * class_counts)
    class_weights /= np.max(class_weights)
    return class_weights.astype(np.float32)

def focal_loss(alpha, gamma=2.0):
    def loss(y_true, y_pred):
        y_true_one_hot = tf.one_hot(y_true, depth=len(alpha))
        alpha_factor = tf.gather(alpha, tf.argmax(y_true_one_hot, axis=-1))
        loss_val = -alpha_factor * y_true_one_hot * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred + 1e-7)
        return tf.reduce_mean(loss_val)
    return loss

# =============================================================================
# 데이터 증강: 노이즈 추가 또는 신호 반전 (flip) 적용
# =============================================================================
def augment_sample(eeg_sample, ppg_sample, noise_std=0.01):
    """
    하나의 EEG/PPG 샘플에 대해 무작위로 가우시안 노이즈 추가 또는 신호 반전을 적용.
    """
    aug_type = np.random.choice([0, 1])
    if aug_type == 0:
        # 가우시안 노이즈 추가
        eeg_aug = eeg_sample + np.random.normal(0, noise_std, size=eeg_sample.shape)
        ppg_aug = ppg_sample + np.random.normal(0, noise_std, size=ppg_sample.shape)
    else:
        # 신호 반전 (부호 반전)
        eeg_aug = -eeg_sample
        ppg_aug = -ppg_sample
    return eeg_aug, ppg_aug

def augment_minority_data(train_eeg, train_ppg, train_labels, noise_std=0.01):
    """
    소수 클래스에 대해 증강(노이즈 추가 또는 신호 반전)하여 다수 클래스와 균형을 맞춥니다.
    각 소수 클래스의 부족 샘플 수 만큼 무작위 증강을 수행합니다.
    """
    unique_classes, counts = np.unique(train_labels, return_counts=True)
    max_count = np.max(counts)
    augmented_eeg = []
    augmented_ppg = []
    augmented_labels = []
    
    for cls in unique_classes:
        indices = np.where(train_labels == cls)[0]
        current_count = len(indices)
        if current_count < max_count:
            deficit = max_count - current_count
            for i in range(deficit):
                idx = np.random.choice(indices)
                eeg_sample = train_eeg[idx]
                ppg_sample = train_ppg[idx]
                eeg_aug, ppg_aug = augment_sample(eeg_sample, ppg_sample, noise_std=noise_std)
                augmented_eeg.append(eeg_aug)
                augmented_ppg.append(ppg_aug)
                augmented_labels.append(cls)
    if len(augmented_eeg) > 0:
        augmented_eeg = np.array(augmented_eeg)
        augmented_ppg = np.array(augmented_ppg)
        augmented_labels = np.array(augmented_labels)
        train_eeg = np.concatenate([train_eeg, augmented_eeg], axis=0)
        train_ppg = np.concatenate([train_ppg, augmented_ppg], axis=0)
        train_labels = np.concatenate([train_labels, augmented_labels], axis=0)
    # 데이터 셔플
    perm = np.random.permutation(train_eeg.shape[0])
    return train_eeg[perm], train_ppg[perm], train_labels[perm]

# =============================================================================
# 8. 데이터 로드 함수
# =============================================================================
def load_multimodal_data(subject):
    """
    subject: 예) "s01", "s02", ..., "s22"
    
    각 피험자별로,
      - EEG 파일: {subject}_eeg_signals_segment_XX.npy, 각 파일 shape: (40, 32, 128)
      - PPG 파일: {subject}_ppg_signals_segment_XX.npy, 각 파일 shape: (40, 128)
      - 라벨 파일: {subject}_three_labels.npy, shape: (40,)
      
    각 npy 파일의 첫 번째 축(40)은 해당 세그먼트 내 샘플 개수를 나타내며,
    각 샘플은 라벨 파일의 인덱스와 1:1 매칭됩니다.
    
    여러 세그먼트 파일을 순차적으로 로드한 후 axis=0으로 concatenate 하여
    최종 데이터셋을 구성합니다.
    
    최종 데이터:
      - EEG: (N, 32, 128, 1)
      - PPG: (N, 128, 1, 1)
      - N = (세그먼트 파일 수) * 40
      - 라벨: 각 세그먼트마다 동일하므로, 라벨을 반복하여 (N,) shape로 구성
    """
    import glob
    eeg_pattern = os.path.join(EEG_DATA_PATH, f"{subject}_eeg_signals_segment_*.npy")
    ppg_pattern = os.path.join(PPG_DATA_PATH, f"{subject}_ppg_signals_segment_*.npy")
    
    eeg_files = sorted(glob.glob(eeg_pattern))
    ppg_files = sorted(glob.glob(ppg_pattern))
    
    if len(eeg_files) == 0 or len(ppg_files) == 0:
        raise FileNotFoundError(f"{subject}에 해당하는 EEG 또는 PPG 파일을 찾을 수 없습니다.")
    
    if len(eeg_files) != len(ppg_files):
        print("경고: EEG와 PPG 파일 수가 일치하지 않습니다.")
    
    eeg_segments = []
    ppg_segments = []
    
    for eeg_file, ppg_file in zip(eeg_files, ppg_files):
        eeg_data = np.load(eeg_file)  # shape: (40, 32, 128)
        eeg_segments.append(eeg_data)
        
        ppg_data = np.load(ppg_file)  # shape: (40, 128)
        ppg_segments.append(ppg_data)
    
    eeg_array = np.concatenate(eeg_segments, axis=0)  # (세그먼트 수*40, 32, 128)
    ppg_array = np.concatenate(ppg_segments, axis=0)  # (세그먼트 수*40, 128)
    
    eeg_array = np.expand_dims(eeg_array, axis=-1)     # (N, 32, 128, 1)
    ppg_array = ppg_array.reshape(ppg_array.shape[0], ppg_array.shape[1], 1, 1)  # (N, 128, 1, 1)
    
    label_file = os.path.join(LABELS_PATH, f"{subject}_three_labels.npy")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"라벨 파일이 없습니다: {label_file}")
    labels = np.load(label_file)  # (40,)
    
    S = len(eeg_files)
    labels_repeated = np.tile(labels, S)  # 최종 라벨 shape: (S*40,)
    
    from sklearn.model_selection import train_test_split
    train_eeg, test_eeg, train_ppg, test_ppg, train_labels, test_labels = train_test_split(
        eeg_array, ppg_array, labels_repeated, test_size=0.2, random_state=42, stratify=labels_repeated)
    
    return train_eeg, train_ppg, train_labels, test_eeg, test_ppg, test_labels

# =============================================================================
# 9. 학습 및 평가
# =============================================================================
def train_multimodal():
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    for subject in subjects:
        print(f"\n==== {subject} 데이터 로드 ====")
        try:
            train_eeg, train_ppg, train_labels, test_eeg, test_ppg, test_labels = load_multimodal_data(subject)
        except Exception as e:
            print(f"{subject} 데이터 로드 실패: {e}")
            continue
        
        # Validation set 구성 (train set의 20%)
        train_eeg, valid_eeg, train_ppg, valid_ppg, train_labels, valid_labels = train_test_split(
            train_eeg, train_ppg, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
        
        # 데이터 증강: 소수 클래스에 대해 노이즈 추가 및 반전 방식 증강 적용
        train_eeg, train_ppg, train_labels = augment_minority_data(train_eeg, train_ppg, train_labels, noise_std=0.01)
        print(f"학습 샘플: EEG {train_eeg.shape}, PPG {train_ppg.shape}, 라벨 {train_labels.shape}")
        
        alpha_values = compute_class_weights(train_labels)
        print(f"alpha values: {alpha_values}")
        
        base_model = build_combined_model(num_classes=3)
        model = MultimodalEmotionClassifier(base_model)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=focal_loss(alpha=alpha_values),
                      metrics=["accuracy"])
        model.summary()
        
        # EarlyStopping 및 ReduceLROnPlateau callback 적용
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        ]
        
        history = model.fit(
            [train_eeg, train_ppg], train_labels,
            validation_data=([valid_eeg, valid_ppg], valid_labels),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2
        )
        
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        model.save_weights(weight_path)
        print(f"✅ 모델 가중치 저장: {weight_path}")
        
        predictions = model.predict([test_eeg, test_ppg])
        inter_pred = predictions[0]
        predicted_labels = np.argmax(inter_pred, axis=-1)
        report = classification_report(test_labels, predicted_labels,
                                       target_names=["Negative", "Positive", "Neutral"],
                                       labels=[0,1,2], zero_division=0)
        print(f"\n📊 {subject} 테스트 리포트\n{report}")
        
        report_path = os.path.join(subject_save_path, f"{subject}_test_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"✅ 테스트 리포트 저장: {report_path}")

if __name__ == "__main__":
    train_multimodal()
