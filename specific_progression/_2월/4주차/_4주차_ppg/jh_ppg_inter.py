import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import mode  # trial별 다수결을 위한 모듈

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, 
                                     Flatten, Dense, Reshape, Concatenate, Lambda)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
# 전역 하이퍼파라미터 설정 (초 단위)
# =============================================================================
WINDOW_SIZE = 10       # 예: 10초짜리 sample
SLIDING_STEP = 1       # 1초 간격 (overlap)

# PPG의 1초 segment 길이 (128 Hz → 128 샘플, 5채널)
PPG_SEGMENT_LENGTH = 128

# 학습 관련 하이퍼파라미터
EPOCHS = 150
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
NUM_CLASSES = 4

# =============================================================================
# PPG 데이터 로딩 및 증강
# 각 파일은 60개의 1초 segment (각 segment의 shape = (PPG_SEGMENT_LENGTH, 5))라고 가정.
# window_size (초)와 SLIDING_STEP (초)를 적용하여 sample 생성:
# sample shape = (PPG_SEGMENT_LENGTH * window_size, 5). 예: (128*10,5) → (1280,5)
# =============================================================================
def load_ppg_data(data_dir='/home/bcml1/2025_EMOTION/DEAP_PPG_1s_nonorm', 
                  ppg_segment_length=PPG_SEGMENT_LENGTH, 
                  window_size=WINDOW_SIZE, 
                  step=SLIDING_STEP):
    subject_data = {}
    file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
    print(f"총 {len(file_paths)}개의 PPG 파일을 찾았습니다.")
    
    for file_path in file_paths:
        base = os.path.basename(file_path)
        try:
            subject_id = base.split('_')[0]
            label_str = base.split('_')[-1].split('.')[0]
            label = int(label_str)
        except Exception as e:
            print("라벨/서브젝트 추출 오류:", file_path, e)
            continue
        data = np.load(file_path)  # 원래 shape: (60, 5, 128) → 60 segment, 5채널, 128 샘플(1초)
        # 만약 (60, 5, 128)라면 transpose하여 (60, 128, 5)
        data = np.transpose(data, (0, 2, 1))
        # 필요한 segment 길이 만큼 슬라이싱 (기본 128)
        data = data[:, :ppg_segment_length, :]
        
        num_segments = data.shape[0]  # 보통 60
        windowed_samples = []
        # sliding window 적용 (1초 단위 overlap)
        for start in range(0, num_segments - window_size + 1, step):
            window = data[start:start+window_size]  # shape: (window_size, ppg_segment_length, 5)
            # time axis로 concatenate → shape: (ppg_segment_length * window_size, 5)
            combined = np.concatenate(window, axis=0)
            windowed_samples.append(combined)
        if not windowed_samples:
            print(f"유효한 sliding window가 없습니다: {file_path}")
            continue
        windowed_samples = np.array(windowed_samples)
        labels = np.full((windowed_samples.shape[0],), label, dtype=np.int32)
        
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(windowed_samples)
        subject_data[subject_id]['y'].append(labels)
    
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        print(f"{subject} - PPG X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
        
        # 오버샘플링으로 클래스 균형 맞추기
        X_data = subject_data[subject]['X']
        y_data = subject_data[subject]['y']
        unique_classes, counts = np.unique(y_data, return_counts=True)
        max_count = counts.max()
        noise_std = 1e-6
        X_aug_list = [X_data]
        y_aug_list = [y_data]
        for cls in unique_classes:
            cls_indices = np.where(y_data == cls)[0]
            if len(cls_indices) < max_count:
                diff = max_count - len(cls_indices)
                sampled_indices = np.random.choice(cls_indices, diff, replace=True)
                X_samples = X_data[sampled_indices]
                noise = np.random.normal(0.0, noise_std, X_samples.shape)
                X_aug = X_samples + noise
                X_aug_list.append(X_aug)
                y_aug_list.append(np.full(diff, cls, dtype=y_data.dtype))
        subject_data[subject]['X'] = np.concatenate(X_aug_list, axis=0)
        subject_data[subject]['y'] = np.concatenate(y_aug_list, axis=0)
        print(f"{subject} after augmentation: PPG X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
    return subject_data

# =============================================================================
# PPG branch용 CNN (Depthwise Separable Conv 구조)
# =============================================================================
def build_cnn_branch(input_length):
    inputs = Input(shape=(input_length, 1))
    x = tf.keras.layers.Conv1D(8, 7, strides=1, padding='same',
                               activation=tf.nn.relu6,
                               kernel_initializer=tf.keras.initializers.HeNormal(),
                               kernel_regularizer=tf.keras.regularizers.l2(1e-6))(inputs)
    x = tf.keras.layers.SeparableConv1D(16, 7, strides=4, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = tf.keras.layers.SeparableConv1D(32, 7, strides=2, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = tf.keras.layers.SeparableConv1D(64, 7, strides=4, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = tf.keras.layers.SeparableConv1D(128, 7, strides=2, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return Model(inputs=inputs, outputs=x)

# =============================================================================
# PPG 특징 추출기: 각 채널별 branch 처리 후 fc layer
# =============================================================================
def create_ppg_feature_extractor(input_shape, num_branches=5):
    ppg_input = Input(shape=input_shape, name='ppg_input_inner')
    branch_outputs = []
    for i in range(num_branches):
        channel = Lambda(lambda x, i=i: x[:, :, i:i+1])(ppg_input)
        branch = build_cnn_branch(input_shape[0])
        branch_outputs.append(branch(channel))
    concatenated = Concatenate()(branch_outputs)
    initializer = tf.keras.initializers.HeNormal()
    reg = tf.keras.regularizers.l2(1e-6)
    x = Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=reg, name='ppg_fc1')(concatenated)
    x = Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=reg, name='ppg_fc2')(x)
    return Model(inputs=ppg_input, outputs=x, name='ppg_feature_extractor')

# =============================================================================
# PPG 단일 모달 모델 생성
# EEG 관련 부분 없이 PPG 특징 추출기 뒤에 분류기 레이어만 구성
# =============================================================================
def create_ppg_model(input_shape, dropout_rate=DROPOUT_RATE, num_classes=NUM_CLASSES):
    ppg_input = Input(shape=input_shape, name='ppg_input')
    features = create_ppg_feature_extractor(input_shape, num_branches=input_shape[-1])(ppg_input)
    initializer = tf.keras.initializers.HeNormal()
    reg = tf.keras.regularizers.l2(1e-6)
    x = Dense(256, activation='relu', kernel_initializer=initializer, kernel_regularizer=reg, name='dense1')(features)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=reg, name='dense2')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inputs=ppg_input, outputs=output)

# =============================================================================
# 단일 모달 (PPG) 데이터 증강: 다중 입력이 아닌 단일 modality용
# =============================================================================
def balance_dataset(X, y, noise_std=0.01):
    y_labels = np.argmax(y, axis=1)
    classes = np.unique(y_labels)
    counts = {cls: np.sum(y_labels == cls) for cls in classes}
    max_count = max(counts.values())
    X_list = [X]
    y_list = [y]
    for cls in classes:
        indices = np.where(y_labels == cls)[0]
        num_needed = max_count - len(indices)
        for _ in range(num_needed):
            idx = np.random.choice(indices)
            sample = X[idx]
            X_aug = sample + np.random.normal(0.0, noise_std, sample.shape)
            X_list.append(X_aug[np.newaxis, ...])
            y_list.append(y[idx][np.newaxis, ...])
    X_augmented = np.concatenate(X_list, axis=0)
    y_augmented = np.concatenate(y_list, axis=0)
    idxs = np.arange(len(y_augmented))
    np.random.shuffle(idxs)
    return X_augmented[idxs], y_augmented[idxs]

# =============================================================================
# 학습 그래프 저장 헬퍼 함수
# =============================================================================
def save_training_history(history, fold_result_dir, subject_id, fold_no):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{subject_id} - Fold {fold_no} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{subject_id} - Fold {fold_no} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fold_result_dir, "training_history.png"))
    plt.close()

# =============================================================================
# leave-one-subject-out 방식 PPG 단일 모달 학습 및 평가
# =============================================================================
def train_ppg_model_leave_one_subject_out(test_subject_id, train_ppg_X, train_y,
                                          test_ppg_X, test_y, test_trial_ids,
                                          num_classes=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                          result_dir_base="/home/bcml1/sigenv/_4주차_ppg/jh_result_inter_seg10s"):
    result_dir = os.path.join(result_dir_base, f"test_{test_subject_id}")
    os.makedirs(result_dir, exist_ok=True)
    
    # training/validation split
    ppg_X_train, ppg_X_val, y_train, y_val = train_test_split(
        train_ppg_X, train_y, test_size=0.2, random_state=42, stratify=np.argmax(train_y, axis=1)
    )
    
    # 데이터 증강 (단일 modality용)
    ppg_X_train, y_train = balance_dataset(ppg_X_train, y_train, noise_std=0.01)
    ppg_X_val, y_val = balance_dataset(ppg_X_val, y_val, noise_std=0.01)
    
    input_shape = train_ppg_X.shape[1:]
    model = create_ppg_model(input_shape, dropout_rate=DROPOUT_RATE, num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint_path = os.path.join(result_dir, "best_model.keras")
    checkpoint_cb = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    
    history = model.fit(ppg_X_train, y_train,
                        validation_data=(ppg_X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[checkpoint_cb, earlystop_cb], verbose=1)
    
    save_training_history(history, result_dir, test_subject_id, fold_no=0)
    
    model.load_weights(checkpoint_path)
    y_pred = model.predict(test_ppg_X)
    
    y_test_sample = np.argmax(test_y, axis=1)
    y_pred_sample = np.argmax(y_pred, axis=1)
    overall_report = classification_report(y_test_sample, y_pred_sample, digits=4)
    overall_cm = confusion_matrix(y_test_sample, y_pred_sample)
    with open(os.path.join(result_dir, "classification_report_overall.txt"), "w") as f:
        f.write(overall_report)
    with open(os.path.join(result_dir, "confusion_matrix_overall.txt"), "w") as f:
        f.write(np.array2string(overall_cm))
    
    aggregated_y_true = []
    aggregated_y_pred = []
    unique_trials = np.unique(test_trial_ids)
    for trial in unique_trials:
        trial_indices = np.where(test_trial_ids == trial)[0]
        if len(trial_indices) == 0:
            continue
        trial_true = np.argmax(test_y[trial_indices][0])
        trial_pred = mode(np.argmax(y_pred[trial_indices], axis=1)).mode.item()
        aggregated_y_true.append(trial_true)
        aggregated_y_pred.append(trial_pred)
    
    trial_report = classification_report(aggregated_y_true, aggregated_y_pred, digits=4)
    trial_cm = confusion_matrix(aggregated_y_true, aggregated_y_pred)
    with open(os.path.join(result_dir, "classification_report_trials.txt"), "w") as f:
        f.write(trial_report)
    with open(os.path.join(result_dir, "confusion_matrix_trials.txt"), "w") as f:
        f.write(np.array2string(trial_cm))
    
    print("=== Sample-level Evaluation ===")
    print(overall_report)
    print("=== Trial-level Evaluation (aggregated via majority vote) ===")
    print(trial_report)
    
    return model, history

# =============================================================================
# Main 함수: PPG 데이터만 로딩 후, leave-one-subject-out 방식으로 단일 모달 학습 및 평가
# =============================================================================
def main():
    ppg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_1s_nonorm"
    
    # PPG 데이터 로딩 (동일한 WINDOW_SIZE, SLIDING_STEP 적용)
    ppg_subject_data = load_ppg_data(ppg_data_dir, ppg_segment_length=PPG_SEGMENT_LENGTH, 
                                     window_size=WINDOW_SIZE, step=SLIDING_STEP)
    
    # ppg_subject_data의 각 key는 subject_id (예: "s01", "s02", ...)
    subjects = sorted(ppg_subject_data.keys())
    subject_data = {}
    for subject in subjects:
        print(f"--- {subject} 데이터 로딩 시작 ---")
        ppg_data = ppg_subject_data[subject]
        ppg_X = ppg_data['X']
        y = ppg_data['y']
        # trial id가 별도로 없다면, sample 개수에 따라 dummy trial id 부여 (또는 실제 trial id가 있다면 그대로 사용)
        trial_ids = np.arange(ppg_X.shape[0])
        subject_data[subject] = {'ppg_X': ppg_X, 'y': to_categorical(y, num_classes=NUM_CLASSES), 'trial_ids': trial_ids}
        print(f"{subject} - PPG data shape: {ppg_X.shape}, y shape: {y.shape}")
    
    result_dir_base = "/home/bcml1/sigenv/_4주차_ppg/jh_result_inter_seg10s"
    os.makedirs(result_dir_base, exist_ok=True)
    
    # Leave-One-Subject-Out 방식 학습
    for test_subject in subject_data.keys():
        print(f"--- Leave-One-Subject-Out: Test subject {test_subject} ---")
        train_ppg_list, train_y_list = [], []
        for subject in subject_data:
            if subject == test_subject:
                continue
            train_ppg_list.append(subject_data[subject]['ppg_X'])
            train_y_list.append(subject_data[subject]['y'])
        train_ppg = np.concatenate(train_ppg_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        test_ppg = subject_data[test_subject]['ppg_X']
        test_y = subject_data[test_subject]['y']
        test_trial_ids = subject_data[test_subject]['trial_ids']
        
        train_ppg_model_leave_one_subject_out(test_subject, train_ppg, train_y,
                                              test_ppg, test_y, test_trial_ids,
                                              num_classes=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                              result_dir_base=result_dir_base)
        
if __name__ == "__main__":
    main()
