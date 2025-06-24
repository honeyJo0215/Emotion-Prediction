#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEAP 데이터셋 (EEG, PPG) 기반 멀티모달 감정 분류 모델
- 각 피험자(s01 ~ s22)는 이미 전처리되어 저장된 데이터 파일을 사용합니다.
- EEG 데이터: (40, 32, 128) → 모델 입력: (40, 32, 128, 1)
- PPG 데이터: (40, 128) → 모델 입력: (40, 128, 1, 1)
- 라벨: (40,) (Negative: 0, Positive: 1, Neutral: 2)
- 데이터 파일은 각 피험자별로 한 파일에 40개의 1초 segment가 저장되어 있습니다.
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
SAVE_PATH    = "//home/bcml1/sigenv/_3주차_eeg+pps_CNN/result1"
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
    eeg_output = Dense(128, activation="relu")(x)
    
    ## PPG Branch
    # PPG 입력: (128, 1, 1)
    ppg_input = Input(shape=(128, 1, 1), name="PPG_Input")
    y = Conv2D(filters=16, kernel_size=(4,1), strides=(2,1), padding='valid', activation='relu')(ppg_input)
    y = DepthwiseConv2D(kernel_size=(6,1), strides=(3,1), padding='valid', activation='relu')(y)
    y = AveragePooling2D(pool_size=(2,1), strides=(2,1))(y)
    y = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), activation='relu')(y)
    y = Flatten()(y)
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
    # 모델 입력은 한 피험자의 40개 segment 데이터
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
        # y가 dict형태이면 Inter_Classification 키 사용
        if isinstance(y, dict):
            y_true = y["Inter_Classification"]
        else:
            y_true = y

        with tf.GradientTape() as tape:
            inter_pred, eeg_pred, ppg_pred, _ = self(x, training=True)
            loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
            loss_eeg   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eeg_pred)
            loss_ppg   = tf.keras.losses.sparse_categorical_crossentropy(y_true, ppg_pred)
            loss = tf.reduce_mean((loss_inter + loss_eeg + loss_ppg) / 3.0)

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
# 7. 오버샘플링 (train set에서 부족한 클래스 복제)
# =============================================================================
def oversample_data(train_eeg, train_ppg, train_labels):
    unique_classes, counts = np.unique(train_labels, return_counts=True)
    max_count = np.max(counts)
    new_eeg, new_ppg, new_labels = [], [], []
    for cls in unique_classes:
        indices = np.where(train_labels == cls)[0]
        rep_factor = int(np.ceil(max_count / len(indices)))
        eeg_cls = train_eeg[indices]
        ppg_cls = train_ppg[indices]
        label_cls = train_labels[indices]
        eeg_rep = np.repeat(eeg_cls, rep_factor, axis=0)
        ppg_rep = np.repeat(ppg_cls, rep_factor, axis=0)
        label_rep = np.repeat(label_cls, rep_factor, axis=0)
        perm = np.random.permutation(eeg_rep.shape[0])
        new_eeg.append(eeg_rep[perm][:max_count])
        new_ppg.append(ppg_rep[perm][:max_count])
        new_labels.append(label_rep[perm][:max_count])
    new_eeg = np.concatenate(new_eeg, axis=0)
    new_ppg = np.concatenate(new_ppg, axis=0)
    new_labels = np.concatenate(new_labels, axis=0)
    perm_all = np.random.permutation(new_eeg.shape[0])
    return new_eeg[perm_all], new_ppg[perm_all], new_labels[perm_all]

# =============================================================================
# 8. 데이터 로드 함수 (각 피험자별 파일에서 40개 segment 불러오기)
# =============================================================================
def load_multimodal_data(subject):
    """
    subject: 예) "s01", "s02", ..., "s22"
    
    각 피험자별로,
      - EEG 파일: {subject}_eeg_signals_segment_XX.npy, 각 파일 shape: (40, 32, 128)
      - PPG 파일: {subject}_ppg_signals_segment_XX.npy, 각 파일 shape: (40, 128)
      - 라벨 파일: {subject}_three_labels.npy, shape: (40,)
      
    여기서 각 npy 파일의 첫 번째 축(40)은 해당 세그먼트 내 샘플 개수를 나타내며,
    각 샘플은 라벨 파일의 인덱스와 1:1 매칭됩니다.
    
    여러 세그먼트 파일에 대해, 각 파일을 순차적으로 load한 후 axis=0으로 concatenate 하여
    최종 데이터셋을 구성합니다.
    
    최종적으로,
      - EEG 데이터: (N, 32, 128, 1)
      - PPG 데이터: (N, 128, 1, 1)
    여기서 N = (세그먼트 파일 수) * 40 이며,  
    각 세그먼트의 40개 샘플에 대해 라벨은 동일하므로, 라벨을 반복하여 (N,) shape로 만듭니다.
    """
    import glob
    # 파일 찾기 (예: s01_eeg_signals_segment_00.npy, s01_eeg_signals_segment_01.npy, ...)
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
    
    # 각 세그먼트 파일을 순차적으로 로드하여 리스트에 추가
    for eeg_file, ppg_file in zip(eeg_files, ppg_files):
        # EEG 파일: (40, 32, 128)
        eeg_data = np.load(eeg_file)
        eeg_segments.append(eeg_data)
        
        # PPG 파일: (40, 128)
        ppg_data = np.load(ppg_file)
        ppg_segments.append(ppg_data)
    
    # 각 modality별로 axis=0으로 연결 → 최종 shape:
    # EEG: (세그먼트 수 * 40, 32, 128), PPG: (세그먼트 수 * 40, 128)
    eeg_array = np.concatenate(eeg_segments, axis=0)
    ppg_array = np.concatenate(ppg_segments, axis=0)
    
    # 채널 차원 추가: EEG → (N, 32, 128, 1), PPG → (N, 128, 1, 1)
    eeg_array = np.expand_dims(eeg_array, axis=-1)
    ppg_array = ppg_array.reshape(ppg_array.shape[0], ppg_array.shape[1], 1, 1)
    
    # 라벨 파일 로드 (shape: (40,))
    label_file = os.path.join(LABELS_PATH, f"{subject}_three_labels.npy")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"라벨 파일이 없습니다: {label_file}")
    labels = np.load(label_file)  # shape: (40,)
    
    # 각 세그먼트 파일마다 40개의 샘플이 있으므로, 세그먼트 수 S:
    S = len(eeg_files)
    # 각 세그먼트의 라벨은 동일하므로, 라벨을 S번 반복하여 최종 라벨 shape: (S*40,)
    labels_repeated = np.tile(labels, S)
    
    # Train/Test 분할 (stratify 적용)
    from sklearn.model_selection import train_test_split
    train_eeg, test_eeg, train_ppg, test_ppg, train_labels, test_labels = train_test_split(
        eeg_array, ppg_array, labels_repeated, test_size=0.2, random_state=42, stratify=labels_repeated)
    
    return train_eeg, train_ppg, train_labels, test_eeg, test_ppg, test_labels

# =============================================================================
# 9. 학습 및 평가
# =============================================================================
def train_multimodal():
    # 피험자: s01 ~ s22
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    for subject in subjects:
        print(f"\n==== {subject} 데이터 로드 ====")
        try:
            train_eeg, train_ppg, train_labels, test_eeg, test_ppg, test_labels = load_multimodal_data(subject)
        except Exception as e:
            print(f"{subject} 데이터 로드 실패: {e}")
            continue
        
        # Train 데이터를 추가로 80:20으로 분할하여 Validation set 구성
        train_eeg, valid_eeg, train_ppg, valid_ppg, train_labels, valid_labels = train_test_split(
            train_eeg, train_ppg, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
        
        # 오버샘플링 적용 (train set)
        train_eeg, train_ppg, train_labels = oversample_data(train_eeg, train_ppg, train_labels)
        print(f"오버샘플링 후 train sample: EEG {train_eeg.shape}, PPG {train_ppg.shape}, 라벨 {train_labels.shape}")
        
        # 클래스 빈도 기반 alpha 계산
        alpha_values = compute_class_weights(train_labels)
        print(f"alpha values: {alpha_values}")
        
        # 모델 생성 및 컴파일
        base_model = build_combined_model(num_classes=3)
        model = MultimodalEmotionClassifier(base_model)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=focal_loss(alpha=alpha_values),
                      metrics=["accuracy"])
        model.summary()
        
        start_epoch = 0
        max_epochs = EPOCHS
        batch_size = BATCH_SIZE
        max_retries = 3
        
        for epoch in range(start_epoch, max_epochs):
            retries = 0
            while retries < max_retries:
                try:
                    print(f"\n🚀 {subject} - Epoch {epoch+1}/{max_epochs} (Retry: {retries})")
                    model.fit(
                        [train_eeg, train_ppg], train_labels,
                        validation_data=([valid_eeg, valid_ppg], valid_labels),
                        epochs=1, batch_size=batch_size, verbose=2
                    )
                    break
                except tf.errors.ResourceExhaustedError:
                    print(f"⚠️ OOM 발생! 재시작 (Retry: {retries+1})...")
                    tf.keras.backend.clear_session()
                    import gc
                    gc.collect()
                    base_model = build_combined_model(num_classes=3)
                    model = MultimodalEmotionClassifier(base_model)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                                  loss=focal_loss(alpha=alpha_values),
                                  metrics=["accuracy"])
                    retries += 1
                    tf.keras.backend.sleep(1)
            else:
                print(f"❌ {subject}의 Epoch {epoch+1}에서 최대 재시도 초과. 다음 피험자로 넘어갑니다.")
                break
        
        # 가중치 저장
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        model.save_weights(weight_path)
        print(f"✅ 모델 가중치 저장: {weight_path}")
        
        # 평가
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
