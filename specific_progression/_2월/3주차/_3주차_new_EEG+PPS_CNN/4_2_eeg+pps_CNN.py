"""
DEAP 데이터셋 (EEG, PPS) 기반 멀티모달 감정 분류 모델
- EEG 데이터: (60, 32, 128) → 모델 입력: (60, 32, 128, 1) (trial당 60개의 1초 세그먼트)
- PPS 데이터: (60, 8, 128) → 모델 입력: (60, 8, 128, 1)
- 라벨: trial별 label (0: Excited, 1: Relaxed, 2: Stressed, 3: Bored)
"""

import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, Dropout, AveragePooling2D, DepthwiseConv2D,
    LayerNormalization, MultiHeadAttention, Input, Lambda, Add, Concatenate, Softmax,
    GlobalAveragePooling1D, TimeDistributed
)
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report

# GPU 메모리 제한 설정
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

# 전처리된 trial 단위 데이터 경로 (예: s01_trial_00_label_0.npy)
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG3"
PPS_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_PPS3"
# 라벨은 파일명에 포함되어 있음 (예: s01_trial_00_label_0.npy 에서 label_0)
LABELS_PATH = "/home/bcml1/2025_EMOTION/DEAP_four_labels"
SAVE_PATH = "/home/bcml1/sigenv/_3주차_new_EEG+PPS_CNN/4_final_1"
os.makedirs(SAVE_PATH, exist_ok=True)


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
                    print(f"\nValidation {self.monitor} did not improve for {self.patience} epochs. Stopping training and restoring best weights.")
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True


# =============================================================================
# 1. Segment-level Feature Extractors
# =============================================================================
def eeg_segment_extractor():
    model = Sequential(name="EEG_Segment_Extractor")
    model.add(Conv2D(filters=16, kernel_size=(4,16), strides=(2,8), padding='valid',
                     activation='relu', input_shape=(32,128,1)))
    model.add(DepthwiseConv2D(kernel_size=(6,6), strides=(3,3), padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation="relu"))
    return model

def pps_segment_extractor():
    model = Sequential(name="PPS_Segment_Extractor")
    model.add(Conv2D(filters=16, kernel_size=(1,16), strides=(1,8), padding='valid',
                     activation='relu', input_shape=(8,128,1)))
    model.add(DepthwiseConv2D(kernel_size=(2,6), strides=(2,3), padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation="relu"))
    return model

# =============================================================================
# 2. Fusion Modules
# =============================================================================
def create_inter_modality_fusion(eeg_features, pps_features, num_heads=4, d_model=128, dropout_rate=0.25):
    # 간단한 cross-modal attention
    eeg_query = Dense(d_model)(eeg_features)
    pps_key_value = Dense(d_model)(pps_features)
    eeg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_query)
    pps_key_value = Lambda(lambda x: tf.expand_dims(x, axis=1))(pps_key_value)
    
    cross_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_Attention")(
        query=eeg_query, key=pps_key_value, value=pps_key_value)
    cross_att = Dropout(dropout_rate)(cross_att)
    cross_att = Add()([eeg_query, cross_att])
    cross_att = LayerNormalization(epsilon=1e-6)(cross_att)
    fused_features = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_att)
    fused_features = Dense(d_model, activation="relu", name="Fused_Linear")(fused_features)
    return fused_features

def create_intra_modality_encoding(feature, num_heads=4, d_model=128, dropout_rate=0.25, name_prefix=""):
    # feature shape: (128,)
    expanded = Lambda(lambda x: tf.expand_dims(x, axis=0))(feature)  # (1,128)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name=name_prefix+"SelfAttention")(
        expanded, expanded, expanded)
    x = Dropout(dropout_rate)(x)
    x = Add()([expanded, x])
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Lambda(lambda x: tf.squeeze(x, axis=0))(x)  # (128,)
    x = Dense(d_model, activation="relu")(x)
    return x

# =============================================================================
# 3. 최종 모델 구성
# =============================================================================
def build_combined_model(num_classes=4):
    # 입력: 각 trial별 EEG: (60,32,128,1), PPS: (60,8,128,1)
    eeg_input = Input(shape=(60,32,128,1), name="EEG_Input")
    pps_input = Input(shape=(60,8,128,1), name="PPS_Input")
    
    # TimeDistributed를 이용해 각 1초 세그먼트별 feature 추출
    td_eeg_features = TimeDistributed(eeg_segment_extractor(), name="TD_EEG_Features")(eeg_input)  # (60,128)
    td_pps_features = TimeDistributed(pps_segment_extractor(), name="TD_PPS_Features")(pps_input)  # (60,128)
    
    # 시간축에 대해 평균 pooling하여 trial-level feature 생성
    eeg_features = GlobalAveragePooling1D(name="GAP_EEG")(td_eeg_features)  # (128,)
    pps_features = GlobalAveragePooling1D(name="GAP_PPS")(td_pps_features)  # (128,)
    
    # Inter-modality fusion
    fused_features = create_inter_modality_fusion(eeg_features, pps_features)
    
    # Intra-modality encoding (각 모달리티별)
    eeg_encoded = create_intra_modality_encoding(eeg_features, name_prefix="EEG_")
    pps_encoded = create_intra_modality_encoding(pps_features, name_prefix="PPS_")
    
    # 분류 branch
    inter_classification = Dense(num_classes, activation="softmax", name="Inter_Classification")(fused_features)
    eeg_classification   = Dense(num_classes, activation="softmax", name="EEG_Classification")(eeg_encoded)
    pps_classification   = Dense(num_classes, activation="softmax", name="PPS_Classification")(pps_encoded)
    
    concat_features = Concatenate(name="Fusion_Concat")([fused_features, eeg_encoded, pps_encoded])
    weights_logits = Dense(units=num_classes, activation=None, name="Weight_Logits")(concat_features)
    weights = Softmax(axis=-1, name="Weight_Softmax")(weights_logits)
    
    model = Model(inputs=[eeg_input, pps_input],
                  outputs=[inter_classification, eeg_classification, pps_classification, weights],
                  name="Multimodal_Emotion_Classifier")
    return model

# =============================================================================
# 4. Custom 학습 모델 클래스 (inter_classification branch만 손실 사용)
# =============================================================================
class MultimodalEmotionClassifier(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super(MultimodalEmotionClassifier, self).__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        # y는 trial별 scalar label
        with tf.GradientTape() as tape:
            inter_pred, _, _, _ = self(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, inter_pred)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, inter_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y = data
        inter_pred, _, _, _ = self(x, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, inter_pred)
        loss = tf.reduce_mean(loss)
        self.compiled_metrics.update_state(y, inter_pred)
        metric_results = {m.name: m.result() for m in self.metrics}
        results = {"accuracy": metric_results.get("accuracy", 0.0), "loss": loss}
        return results

# =============================================================================
# 5. 데이터 증강 및 밸런싱 함수 (기존 함수 사용)
# =============================================================================
def augment_sample(eeg_sample, pps_sample, noise_std_eeg=0.01, noise_std_pps=0.005):
    aug_type = np.random.choice([0, 1])
    if aug_type == 0:
        eeg_aug = eeg_sample + np.random.normal(0, noise_std_eeg, size=eeg_sample.shape)
        pps_aug = pps_sample + np.random.normal(0, noise_std_pps, size=pps_sample.shape)
    else:
        eeg_aug = -eeg_sample
        pps_aug = -pps_sample
    return eeg_aug, pps_aug

def balance_data(train_eeg, train_pps, train_labels):
    unique, counts = np.unique(train_labels, return_counts=True)
    max_count = np.max(counts)
    balanced_eeg, balanced_pps, balanced_labels = [], [], []
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
    N = train_eeg.shape[0]
    aug_eeg, aug_pps, aug_labels = [], [], []
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
# 6. 데이터 로드 함수 (trial 단위 데이터 불러오기)
# =============================================================================
def load_multimodal_data(subject):
    # 파일명 패턴 예: s01_trial_00_label_0.npy
    eeg_pattern = os.path.join(EEG_DATA_PATH, f"{subject}_trial_*_label_*.npy")
    pps_pattern = os.path.join(PPS_DATA_PATH, f"{subject}_trial_*_label_*.npy")
    
    eeg_files = sorted(glob.glob(eeg_pattern))
    pps_files = sorted(glob.glob(pps_pattern))
    
    if len(eeg_files) == 0 or len(pps_files) == 0:
        raise FileNotFoundError(f"{subject}에 해당하는 EEG 또는 PPS 파일을 찾을 수 없습니다.")
    
    eeg_list, pps_list, labels_list = [], [], []
    for file in eeg_files:
        data = np.load(file)  # shape: (60,32,128)
        data = np.expand_dims(data, axis=-1)  # (60,32,128,1)
        eeg_list.append(data)
        base = os.path.basename(file)
        parts = base.split("_")
        label_str = parts[-1].split(".")[0]  # 예: "0"
        labels_list.append(int(label_str))
    for file in pps_files:
        data = np.load(file)  # shape: (60,8,128)
        data = np.expand_dims(data, axis=-1)  # (60,8,128,1)
        pps_list.append(data)
    
    eeg_array = np.stack(eeg_list, axis=0)   # (num_trials, 60,32,128,1)
    pps_array = np.stack(pps_list, axis=0)     # (num_trials, 60,8,128,1)
    labels_array = np.array(labels_list)       # (num_trials,)
    
    train_eeg, test_eeg, train_pps, test_pps, train_labels, test_labels = train_test_split(
        eeg_array, pps_array, labels_array, test_size=0.2, random_state=42, stratify=labels_array)
    return train_eeg, train_pps, train_labels, test_eeg, test_pps, test_labels

# =============================================================================
# 7. 학습 history 플롯 저장 함수
# =============================================================================
def plot_and_save_history(history, subject, save_dir):
    plt.figure()
    plt.plot(history.history.get('accuracy', history.history.get('acc')), label='Train Accuracy')
    plt.plot(history.history.get('val_accuracy', history.history.get('val_acc')), label='Validation Accuracy')
    plt.title(f"{subject} Final Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{subject}_final_accuracy.png"))
    plt.close()
    
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
# 8. 최종 학습 및 평가 (10-fold CV로 best weight 확보 후 전체 train 데이터를 대상으로 150 에폭 학습)
#     - 최종 학습에서는 test 데이터는 분리하고, 학습/validation curve만 기록합니다.
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

        print(f"{subject} - Train EEG shape: {train_eeg.shape}, PPS shape: {train_pps.shape}, Labels shape: {labels.shape}")
        
        # 10-fold CV로 best weight 확보
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        best_val_acc = -np.Inf
        best_weights = None
        
        for train_index, val_index in skf.split(train_eeg, labels):
            fold_train_eeg, fold_valid_eeg = train_eeg[train_index], train_eeg[val_index]
            fold_train_pps, fold_valid_pps = train_pps[train_index], train_pps[val_index]
            fold_train_labels, fold_valid_labels = labels[train_index], labels[val_index]
            
            fold_train_eeg, fold_train_pps, fold_train_labels = balance_data(fold_train_eeg, fold_train_pps, fold_train_labels)
            fold_train_eeg, fold_train_pps, fold_train_labels = augment_all_data(
                fold_train_eeg, fold_train_pps, fold_train_labels,
                noise_std_eeg=0.01, noise_std_pps=0.005, factor=5)
            
            base_model = build_combined_model(num_classes=4)
            model = MultimodalEmotionClassifier(base_model)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=["accuracy"])
            
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
            # 키가 'val_accuracy' 또는 'val_acc'로 기록될 수 있으므로 확인
            if 'val_accuracy' in history_cv.history:
                val_key = 'val_accuracy'
            elif 'val_acc' in history_cv.history:
                val_key = 'val_acc'
            else:
                raise ValueError("No validation accuracy key found in history.")
            current_val_acc = max(history_cv.history[val_key])
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_weights = model.get_weights()
        
        print(f"{subject} best validation accuracy from CV: {best_val_acc:.4f}")
        
        # 최종 학습: 전체 train 데이터 대상으로 validation_split=0.2 사용, 150 에폭 학습 (early stopping 없이)
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
        
        # 최종 학습 후 모델 가중치 저장
        weight_path = os.path.join(subject_save_dir, f"{subject}_final_multimodal_model.weights.h5")
        model_final.save_weights(weight_path)
        print(f"✅ {subject} 최종 모델 가중치 저장: {weight_path}")
        
        # 최종 test 평가 (예측 및 classification report 출력)
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
