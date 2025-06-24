"""
DEAP 데이터셋 (EEG, PPS) 기반 멀티모달 감정 분류 모델
- EEG 데이터: (40, 32, 128) → 모델 입력: (40, 32, 128, 1)
- PPS 데이터: (40, 8, 128) → 모델 입력: (40, 8, 128, 1)
- 라벨: (40,) (0: Excited, 1: Relaxed, 2: Stressed, 3: Bored)
- 각 파일에는 40개의 1초 샘플이 저장되어 있으며, 이들 샘플은 라벨 파일과 1:1 매칭됩니다.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, Dropout, AveragePooling2D, DepthwiseConv2D,
    LayerNormalization, MultiHeadAttention, Input, Lambda, Add, Concatenate, Softmax
)
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report

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
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-4

EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG2"
PPS_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_PPS2"
LABELS_PATH  = "/home/bcml1/2025_EMOTION/DEAP_four_labels"
SAVE_PATH    = "/home/bcml1/sigenv/_3주차_new_EEG+PPS_CNN/4_result1"
os.makedirs(SAVE_PATH, exist_ok=True)

# =============================================================================
# 1. Dual‑Stream Feature Extractor (EEG & PPS)
# =============================================================================
def create_dual_stream_feature_extractor():
    """
    EEG와 PPS를 각각 CNN으로 특징 추출
    - EEG Branch: 입력 (32, 128, 1)
    - PPS Branch: 입력 (8, 128, 1)
    """
    ## EEG Branch
    eeg_input = Input(shape=(32, 128, 1), name="EEG_Input")
    x = Conv2D(filters=16, kernel_size=(4,16), strides=(2,8), padding='valid', activation='relu')(eeg_input)
    x = DepthwiseConv2D(kernel_size=(6,6), strides=(3,3), padding='valid', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    eeg_output = Dense(128, activation="relu")(x)
    
    ## PPS Branch
    pps_input = Input(shape=(8, 128, 1), name="PPS_Input")
    y = Conv2D(filters=16, kernel_size=(1,16), strides=(1,8), padding='valid', activation='relu')(pps_input)
    y = DepthwiseConv2D(kernel_size=(2,6), strides=(2,3), padding='valid', activation='relu')(y)
    y = AveragePooling2D(pool_size=(2,2), strides=(2,2))(y)
    y = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), activation='relu')(y)
    y = Flatten()(y)
    y = Dropout(0.25)(y)
    pps_output = Dense(128, activation="relu")(y)
    
    model = Model(inputs=[eeg_input, pps_input], outputs=[eeg_output, pps_output],
                  name="DualStreamFeatureExtractor")
    return model

# =============================================================================
# 2. Inter‑Modality Fusion Module (Cross‑Modal Transformer)
# =============================================================================
def create_inter_modality_fusion(eeg_features, pps_features, num_heads=4, d_model=128, dropout_rate=0.25):
    """
    EEG와 PPS 간의 상호관계를 학습하는 Cross‑Modal Transformer
    """
    # EEG → PPS
    eeg_query = Dense(d_model)(eeg_features)
    pps_key_value = Dense(d_model)(pps_features)
    eeg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_query)
    pps_key_value = Lambda(lambda x: tf.expand_dims(x, axis=1))(pps_key_value)
    
    cross_att_1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_EEG_to_PPS")(
        query=eeg_query, key=pps_key_value, value=pps_key_value)
    cross_att_1 = Dropout(dropout_rate)(cross_att_1)
    cross_att_1 = Add()([eeg_query, cross_att_1])
    cross_att_1 = LayerNormalization(epsilon=1e-6)(cross_att_1)
    cross_att_1 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_att_1)
    
    # PPS → EEG
    pps_query = Dense(d_model)(pps_features)
    pps_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(pps_query)
    eeg_key_value_2 = Dense(d_model)(eeg_features)
    eeg_key_value_2 = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_key_value_2)
    
    cross_att_2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_PPS_to_EEG")(
        query=pps_query, key=eeg_key_value_2, value=eeg_key_value_2)
    cross_att_2 = Dropout(dropout_rate)(cross_att_2)
    cross_att_2 = Add()([pps_query, cross_att_2])
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
# 3. Intra‑Modality Encoding Module
# =============================================================================
def create_intra_modality_encoding(eeg_features, pps_features, num_heads=4, d_model=128, dropout_rate=0.25):
    """
    EEG와 PPS 각각의 고유 특성을 보존 및 강화하는 Intra‑Modal Encoding
    """
    # EEG Self-Attention
    eeg_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_features)
    eeg_self_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_EEG")(
        query=eeg_expanded, key=eeg_expanded, value=eeg_expanded)
    eeg_self_att = Dropout(dropout_rate)(eeg_self_att)
    eeg_self_att = Add()([eeg_expanded, eeg_self_att])
    eeg_self_att = LayerNormalization(epsilon=1e-6)(eeg_self_att)
    eeg_self_att = Lambda(lambda x: tf.squeeze(x, axis=1))(eeg_self_att)
    
    # PPS Self-Attention
    pps_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(pps_features)
    pps_self_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_PPS")(
        query=pps_expanded, key=pps_expanded, value=pps_expanded)
    pps_self_att = Dropout(dropout_rate)(pps_self_att)
    pps_self_att = Add()([pps_expanded, pps_self_att])
    pps_self_att = LayerNormalization(epsilon=1e-6)(pps_self_att)
    pps_self_att = Lambda(lambda x: tf.squeeze(x, axis=1))(pps_self_att)
    
    return eeg_self_att, pps_self_att

# =============================================================================
# 4. 전체 모델 구성 (EEG, PPS 입력 → Dual‑Stream Extraction → Fusion → 분류)
# =============================================================================
def build_combined_model(num_classes=4):
    """
    EEG와 PPS 데이터를 받아서  
      - Dual‑Stream Feature Extraction  
      - Cross‑Modal Fusion  
      - Intra‑Modal Encoding  
    을 거쳐 최종 분류 결과를 출력하는 모델 구성
    
    출력 분류: 4가지 감정 상태 (0: Excited, 1: Relaxed, 2: Stressed, 3: Bored)
    """
    eeg_input = Input(shape=(32, 128, 1), name="EEG_Input")
    pps_input = Input(shape=(8, 128, 1), name="PPS_Input")
    
    dual_extractor = create_dual_stream_feature_extractor()
    eeg_features, pps_features = dual_extractor([eeg_input, pps_input])
    
    fused_features = create_inter_modality_fusion(eeg_features, pps_features)
    eeg_encoded, pps_encoded = create_intra_modality_encoding(eeg_features, pps_features)
    
    inter_classification = Dense(num_classes, activation="softmax", name="Inter_Classification")(fused_features)
    eeg_classification   = Dense(num_classes, activation="softmax", name="EEG_Classification")(eeg_encoded)
    pps_classification   = Dense(num_classes, activation="softmax", name="PPS_Classification")(pps_encoded)
    
    concat_features = Concatenate()([fused_features, eeg_encoded, pps_encoded])
    weights_logits = Dense(units=4, activation=None, name="Weight_Logits")(concat_features)
    weights = Softmax(axis=-1, name="Weight_Softmax")(weights_logits)
    
    model = Model(inputs=[eeg_input, pps_input],
                  outputs=[inter_classification, eeg_classification, pps_classification, weights],
                  name="Multimodal_Emotion_Classifier")
    return model

# =============================================================================
# 5. Custom 학습 단계 포함 모델 클래스 정의
# 
# (여기서는 주로 inter_classification branch의 손실만 사용하도록 수정)
# =============================================================================
class MultimodalEmotionClassifier(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super(MultimodalEmotionClassifier, self).__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        y_true = y if not isinstance(y, dict) else y["Inter_Classification"]

        with tf.GradientTape() as tape:
            # 오직 inter_classification branch의 출력만 사용하여 손실 계산
            inter_pred, _, _, _ = self(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
            loss = tf.reduce_mean(loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y_true, inter_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y = data
        y_true = y if not isinstance(y, dict) else y["Inter_Classification"]
        inter_pred, _, _, _ = self(x, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
        loss = tf.reduce_mean(loss)
        self.compiled_metrics.update_state(y_true, inter_pred)
        metric_results = {m.name: m.result() for m in self.metrics}
        results = {"accuracy": metric_results.get("accuracy", 0.0), "loss": loss}
        return results

# =============================================================================
# 6. (Optional) 클래스 빈도 기반 가중치 계산 (현재 사용하지 않음)
# =============================================================================
def compute_class_weights(labels, num_classes=4):
    class_counts = np.bincount(labels, minlength=num_classes)
    total = np.sum(class_counts)
    class_weights = total / (num_classes * class_counts)
    class_weights /= np.max(class_weights)
    return class_weights.astype(np.float32)

# =============================================================================
# 7. 데이터 증강 함수 (EEG: noise_std=0.01, PPS: noise_std=0.005)
# =============================================================================
def augment_sample(eeg_sample, pps_sample, noise_std_eeg=0.01, noise_std_pps=0.005):
    """
    하나의 EEG/PPS 샘플에 대해 무작위로 가우시안 노이즈 추가 또는 신호 반전을 적용.
    """
    aug_type = np.random.choice([0, 1])
    if aug_type == 0:
        eeg_aug = eeg_sample + np.random.normal(0, noise_std_eeg, size=eeg_sample.shape)
        pps_aug = pps_sample + np.random.normal(0, noise_std_pps, size=pps_sample.shape)
    else:
        eeg_aug = -eeg_sample
        pps_aug = -pps_sample
    return eeg_aug, pps_aug

def balance_data(train_eeg, train_pps, train_labels):
    """
    원본 데이터셋에서 각 클래스의 수를 동일하게 맞추기 위해 oversampling (with replacement) 수행.
    """
    unique, counts = np.unique(train_labels, return_counts=True)
    max_count = np.max(counts)
    balanced_eeg = []
    balanced_pps = []
    balanced_labels = []
    for cls in unique:
        indices = np.where(train_labels == cls)[0]
        if len(indices) < max_count:
            oversampled = np.random.choice(indices, size=max_count, replace=True)
        else:
            oversampled = indices
        balanced_eeg.append(train_eeg[oversampled])
        balanced_pps.append(train_pps[oversampled])
        balanced_labels.append(train_labels[oversampled])
    balanced_eeg = np.concatenate(balanced_eeg, axis=0)
    balanced_pps = np.concatenate(balanced_pps, axis=0)
    balanced_labels = np.concatenate(balanced_labels, axis=0)
    perm = np.random.permutation(balanced_labels.shape[0])
    return balanced_eeg[perm], balanced_pps[perm], balanced_labels[perm]

def augment_all_data(train_eeg, train_pps, train_labels, noise_std_eeg=0.01, noise_std_pps=0.005, factor=5):
    """
    각 샘플마다 (factor-1)개의 증강 샘플 생성하여 원본 데이터와 함께 총 factor배 데이터 생성.
    """
    N = train_eeg.shape[0]
    aug_eeg = []
    aug_pps = []
    aug_labels = []
    for i in range(N):
        for _ in range(factor - 1):
            e_aug, p_aug = augment_sample(train_eeg[i], train_pps[i], noise_std_eeg, noise_std_pps)
            aug_eeg.append(e_aug)
            aug_pps.append(p_aug)
            aug_labels.append(train_labels[i])
    aug_eeg = np.array(aug_eeg)
    aug_pps = np.array(aug_pps)
    aug_labels = np.array(aug_labels)
    all_eeg = np.concatenate([train_eeg, aug_eeg], axis=0)
    all_pps = np.concatenate([train_pps, aug_pps], axis=0)
    all_labels = np.concatenate([train_labels, aug_labels], axis=0)
    perm = np.random.permutation(all_labels.shape[0])
    return all_eeg[perm], all_pps[perm], all_labels[perm]

# =============================================================================
# 8. 데이터 로드 함수
# =============================================================================
def load_multimodal_data(subject):
    """
    subject: 예) "s01", "s02", ..., "s22"
    
    각 피험자별로,
      - EEG 파일: {subject}_eeg_signals_segment_XX.npy, shape: (40, 32, 128)
      - PPS 파일: {subject}_pps_signals_segment_XX.npy, shape: (40, 8, 128)
      - 라벨 파일: {subject}_emotion_labels.npy, shape: (40,)
      
    여러 세그먼트 파일을 순차적으로 로드하여 전체 데이터셋을 구성합니다.
    최종 데이터:
      - EEG: (N, 32, 128, 1)
      - PPS: (N, 8, 128, 1)
      - N = (세그먼트 파일 수) * 40
      - 라벨: 각 세그먼트마다 동일하므로 (N,) shape
    """
    import glob
    eeg_pattern = os.path.join(EEG_DATA_PATH, f"{subject}_eeg_signals_segment_*.npy")
    pps_pattern = os.path.join(PPS_DATA_PATH, f"{subject}_pps_signals_segment_*.npy")
    
    eeg_files = sorted(glob.glob(eeg_pattern))
    pps_files = sorted(glob.glob(pps_pattern))
    
    if len(eeg_files) == 0 or len(pps_files) == 0:
        raise FileNotFoundError(f"{subject}에 해당하는 EEG 또는 PPS 파일을 찾을 수 없습니다.")
    
    if len(eeg_files) != len(pps_files):
        print("경고: EEG와 PPS 파일 수가 일치하지 않습니다.")
    
    eeg_segments = []
    pps_segments = []
    
    for eeg_file, pps_file in zip(eeg_files, pps_files):
        eeg_data = np.load(eeg_file)  # shape: (40, 32, 128)
        eeg_segments.append(eeg_data)
        pps_data = np.load(pps_file)  # shape: (40, 8, 128)
        pps_segments.append(pps_data)
    
    eeg_array = np.concatenate(eeg_segments, axis=0)  # (세그먼트 수*40, 32, 128)
    pps_array = np.concatenate(pps_segments, axis=0)    # (세그먼트 수*40, 8, 128)
    
    eeg_array = np.expand_dims(eeg_array, axis=-1)       # (N, 32, 128, 1)
    pps_array = np.expand_dims(pps_array, axis=-1)         # (N, 8, 128, 1)
    
    # 라벨 파일: {subject}_emotion_labels.npy (shape: (40,))
    label_file = os.path.join(LABELS_PATH, f"{subject}_emotion_labels.npy")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"라벨 파일이 없습니다: {label_file}")
    labels = np.load(label_file)  # (40,)
    
    S = len(eeg_files)
    labels_repeated = np.tile(labels, S)  # (S*40,)
    
    # train/test 분할 (80% train, 20% test)
    train_eeg, test_eeg, train_pps, test_pps, train_labels, test_labels = train_test_split(
        eeg_array, pps_array, labels_repeated, test_size=0.2, random_state=42, stratify=labels_repeated)
    
    return train_eeg, train_pps, train_labels, test_eeg, test_pps, test_labels

# =============================================================================
# 9. 커스텀 콜백: validation accuracy가 7 에포크 동안 개선되지 않으면 최고 가중치 복원
# =============================================================================
class RestoreBestWeightsOnDrop(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_accuracy', min_delta=0.001, patience=7, verbose=1):
        super(RestoreBestWeightsOnDrop, self).__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.best = -np.Inf
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        if current > self.best + self.min_delta:
            self.best = current
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"\nValidation {self.monitor} did not improve for {self.patience} epochs. Restoring best weights.")
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True

# =============================================================================
# 학습 history (accuracy, loss) 플롯 저장 함수
# =============================================================================
def plot_and_save_history(history, subject, save_dir):
    # Accuracy curve
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{subject} Final Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{subject}_final_accuracy.png"))
    plt.close()
    
    # Loss curve
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{subject} Final Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{subject}_final_loss.png"))
    plt.close()

# =============================================================================
# 10. 최종 학습 및 평가 (논문 방식)
#     - 먼저 10-fold CV로 best weight 확보 후, 해당 가중치를 초기값으로 하여
#       전체 데이터를 대상으로 train/validation split (validation_split=0.2)으로
#       150 에폭 최종 학습 (early stopping 없이) 진행.
#     - test 데이터는 따로 평가하여 classification report를 출력 및 저장합니다.
# =============================================================================
def train_multimodal_final():
    from sklearn.model_selection import StratifiedKFold
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    for subject in subjects:
        print(f"\n==== {subject} 데이터 로드 ====")
        try:
            train_eeg, train_pps, labels, test_eeg, test_pps, test_labels = load_multimodal_data(subject)
        except Exception as e:
            print(f"{subject} 데이터 로드 실패: {e}")
            continue

        print(f"EEG shape: {train_eeg.shape}, PPS shape: {train_pps.shape}, Labels shape: {labels.shape}")
        
        # 10-fold CV를 통해 best weight 확보
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        best_val_acc = -np.Inf
        best_weights = None
        
        for train_index, val_index in skf.split(train_eeg, labels):
            fold_train_eeg, fold_valid_eeg = train_eeg[train_index], train_eeg[val_index]
            fold_train_pps, fold_valid_pps = train_pps[train_index], train_pps[val_index]
            fold_train_labels, fold_valid_labels = labels[train_index], labels[val_index]
            
            # 데이터 밸런싱 및 증강
            fold_train_eeg, fold_train_pps, fold_train_labels = balance_data(fold_train_eeg, fold_train_pps, fold_train_labels)
            fold_train_eeg, fold_train_pps, fold_train_labels = augment_all_data(
                fold_train_eeg, fold_train_pps, fold_train_labels, noise_std_eeg=0.01, noise_std_pps=0.005, factor=5)
            
            base_model = build_combined_model(num_classes=4)
            model = MultimodalEmotionClassifier(base_model)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=["accuracy"])
            
            # CV fold 학습 (early stopping 사용)
            cv_callbacks = [
                RestoreBestWeightsOnDrop(monitor='val_accuracy', min_delta=0.001, patience=7, verbose=0),
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, min_delta=0.001, restore_best_weights=True, verbose=0),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=0)
            ]
            
            history_cv = model.fit(
                [fold_train_eeg, fold_train_pps], fold_train_labels,
                validation_data=([fold_valid_eeg, fold_valid_pps], fold_valid_labels),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=cv_callbacks,
                verbose=0
            )
            
            val_acc = max(history_cv.history['val_accuracy'])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = model.get_weights()
        
        print(f"{subject} best validation accuracy from CV: {best_val_acc:.4f}")
        
        # 최종 학습: 전체 데이터를 대상으로 validation_split=0.2로 150 에폭 학습 (early stopping 없이)
        final_model = build_combined_model(num_classes=4)
        final_model.set_weights(best_weights)
        model_final = MultimodalEmotionClassifier(final_model)
        model_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                             metrics=["accuracy"])
        
        subject_save_dir = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_dir, exist_ok=True)
        
        history_final = model_final.fit(
            [train_eeg, train_pps], labels,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        
        # 학습 history 플롯 저장
        plot_and_save_history(history_final, subject, subject_save_dir)
        
        # 최종 모델 가중치 저장
        weight_path = os.path.join(subject_save_dir, f"{subject}_final_multimodal_model.weights.h5")
        model_final.save_weights(weight_path)
        print(f"✅ {subject} 최종 모델 가중치 저장: {weight_path}")
        
        # 최종 학습 요약 저장 (마지막 epoch의 train/val 결과)
        summary_file = os.path.join(subject_save_dir, f"{subject}_final_summary.txt")
        final_train_acc = history_final.history['accuracy'][-1]
        final_val_acc = history_final.history['val_accuracy'][-1]
        final_train_loss = history_final.history['loss'][-1]
        final_val_loss = history_final.history['val_loss'][-1]
        with open(summary_file, "w") as f:
            f.write(f"최종 학습 마지막 epoch - Train Accuracy: {final_train_acc:.4f}, Validation Accuracy: {final_val_acc:.4f}\n")
            f.write(f"최종 학습 마지막 epoch - Train Loss: {final_train_loss:.4f}, Validation Loss: {final_val_loss:.4f}\n")
        print(f"✅ {subject} 최종 학습 요약 저장: {summary_file}")
        
        # 최종 학습 후, test 데이터에 대해 classification report 생성
        predictions = model_final.predict([test_eeg, test_pps])
        inter_pred = predictions[0]
        predicted_labels = np.argmax(inter_pred, axis=-1)
        report = classification_report(test_labels, predicted_labels,
                                         target_names=["Excited", "Relaxed", "Stressed", "Bored"],
                                         labels=[0,1,2,3], zero_division=0)
        print(f"\n📊 {subject} 테스트 리포트\n{report}")
        report_path = os.path.join(subject_save_dir, f"{subject}_test_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"✅ {subject} 테스트 리포트 저장: {report_path}")

if __name__ == "__main__":
    train_multimodal_final()
