import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import mode  # trial별 다수결을 위한 모듈
from scipy.signal import butter, filtfilt, detrend  # 필터링 및 기저선 제거용
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, 
                                     Flatten, Dense, Reshape, Concatenate, Lambda, GlobalAveragePooling1D)
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

# PPG의 1초 segment 길이 (128 Hz → 128 샘플)
PPG_SEGMENT_LENGTH = 128

# 학습 관련 하이퍼파라미터
EPOCHS = 150
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
NUM_CLASSES = 4

# =============================================================================
# 전처리 함수들: 밴드패스 필터, 스무딩, 기저선 제거, 정규화
# =============================================================================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    butterworth 밴드패스 필터를 적용하여 lowcut ~ highcut Hz 대역만 남깁니다.
    data: (n_samples, ) 1차원 배열
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def moving_average(data, window_size=3):
    """
    1D 신호에 대해 간단한 이동평균 스무딩을 적용합니다.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def preprocess_ppg_trial(trial_data, fs=128):
    """
    trial_data: (num_segments, segment_length, num_channels)
    각 segment에 대해 기저선 제거, 밴드패스 필터, 스무딩, 정규화를 적용합니다.
    예) PPG의 유의 주파수 대역: 0.5 ~ 8 Hz
    """
    processed_segments = []
    for segment in trial_data:
        # segment shape: (segment_length, num_channels)
        # 1. 기저선 제거 (detrend)
        detrended = detrend(segment, axis=0)
        # 2. 채널별로 밴드패스 필터 적용 (여기서는 0.5 ~ 8 Hz)
        filtered = np.zeros_like(detrended)
        for ch in range(detrended.shape[1]):
            filtered[:, ch] = bandpass_filter(detrended[:, ch], lowcut=0.5, highcut=8.0, fs=fs, order=4)
        # 3. 스무딩: 각 채널에 대해 이동 평균 적용
        smoothed = np.zeros_like(filtered)
        for ch in range(filtered.shape[1]):
            smoothed[:, ch] = moving_average(filtered[:, ch], window_size=3)
        # 4. 정규화: Z-스코어 (채널별)
        normalized = (smoothed - np.mean(smoothed, axis=0)) / (np.std(smoothed, axis=0) + 1e-8)
        processed_segments.append(normalized)
    processed_data = np.array(processed_segments)
    return processed_data

# =============================================================================
# SE Block: 채널 간 중요도 재조정을 통해 특징 표현을 강화 (Hu et al., 2018)
# =============================================================================
def se_block(input_tensor, reduction=4):
    filters = int(input_tensor.shape[-1])
    se = GlobalAveragePooling1D()(input_tensor)
    se = Reshape((1, filters))(se)
    se = Dense(filters // reduction, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return input_tensor * se

# =============================================================================
# PPG 데이터 로딩 및 증강
# 각 파일명은 예: s01_trial_00_label_0.npy 와 같이 되어 있으며,
# sliding window를 적용해 sample을 생성하고, 각 sample에 해당 trial id를 부여합니다.
# sample shape = (PPG_SEGMENT_LENGTH * WINDOW_SIZE, 5) → (1280,5)
# =============================================================================
def load_ppg_data(data_dir='/home/bcml1/2025_EMOTION/DEAP_PPG_1s', 
                  ppg_segment_length=PPG_SEGMENT_LENGTH, 
                  window_size=WINDOW_SIZE, 
                  step=SLIDING_STEP):
    subject_data = {}
    file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
    print(f"총 {len(file_paths)}개의 PPG 파일을 찾았습니다.")
    pattern = r"(s\d+)_trial_(\d+)_label_(\d+)\.npy"
    
    for file_path in file_paths:
        base = os.path.basename(file_path)
        match = re.match(pattern, base)
        if not match:
            print(f"파일명 패턴 불일치: {base}")
            continue
        subject_id = match.group(1)       # 예: s01
        trial_str = match.group(2)          # 예: 00
        trial_id = int(trial_str)           # 정수형 trial id
        label = int(match.group(3))
        
        data = np.load(file_path)  # 원래 shape: (60, 5, 128)
        # 만약 (60, 5, 128)라면 transpose하여 (60, 128, 5)
        data = np.transpose(data, (0, 2, 1))
        data = data[:, :ppg_segment_length, :]  # 필요한 길이 만큼 슬라이싱
        
        # ===== 전처리 추가: 각 trial의 모든 segment에 대해 전처리 수행 =====
        data = preprocess_ppg_trial(data, fs=128)
        
        num_segments = data.shape[0]  # 보통 60
        windowed_samples = []
        trial_ids_list = []
        # sliding window 적용 (1초 간격)
        for start in range(0, num_segments - window_size + 1, step):
            window = data[start:start+window_size]  # shape: (window_size, ppg_segment_length, 5)
            combined = np.concatenate(window, axis=0)  # (1280, 5)
            windowed_samples.append(combined)
            trial_ids_list.append(trial_id)  # 해당 파일의 trial id를 부여
        
        if not windowed_samples:
            print(f"유효한 sliding window가 없습니다: {file_path}")
            continue
        
        windowed_samples = np.array(windowed_samples)
        labels_array = np.full((windowed_samples.shape[0],), label, dtype=np.int32)
        
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': [], 'trial_ids': []}
        subject_data[subject_id]['X'].append(windowed_samples)
        subject_data[subject_id]['y'].append(labels_array)
        subject_data[subject_id]['trial_ids'].append(np.array(trial_ids_list))
    
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        subject_data[subject]['trial_ids'] = np.concatenate(subject_data[subject]['trial_ids'], axis=0)
        print(f"{subject} - PPG X shape: {subject_data[subject]['X'].shape}, "
              f"y shape: {subject_data[subject]['y'].shape}, "
              f"trial_ids shape: {subject_data[subject]['trial_ids'].shape}")
        
        # 오버샘플링으로 클래스 균형 맞추기
        X_data = subject_data[subject]['X']
        y_data = subject_data[subject]['y']
        unique_classes, counts = np.unique(y_data, return_counts=True)
        max_count = counts.max()
        noise_std = 1e-6
        X_aug_list = [X_data]
        y_aug_list = [y_data]
        trial_ids_list = [subject_data[subject]['trial_ids']]
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
                trial_ids_list.append(subject_data[subject]['trial_ids'][sampled_indices])
        subject_data[subject]['X'] = np.concatenate(X_aug_list, axis=0)
        subject_data[subject]['y'] = np.concatenate(y_aug_list, axis=0)
        subject_data[subject]['trial_ids'] = np.concatenate(trial_ids_list, axis=0)
        print(f"{subject} after augmentation: PPG X shape: {subject_data[subject]['X'].shape}, "
              f"y shape: {subject_data[subject]['y'].shape}, "
              f"trial_ids shape: {subject_data[subject]['trial_ids'].shape}")
    return subject_data

# =============================================================================
# PPG branch용 CNN (Depthwise Separable Conv 구조) 개선:
# - 각 convolution 후 BatchNormalization 추가
# - 마지막 convolution 이후 SE block 추가
# =============================================================================
def build_cnn_branch(input_length):
    inputs = Input(shape=(input_length, 1))
    x = tf.keras.layers.Conv1D(8, 7, strides=1, padding='same',
                               activation=tf.nn.relu6,
                               kernel_initializer=tf.keras.initializers.HeNormal(),
                               kernel_regularizer=tf.keras.regularizers.l2(1e-6))(inputs)
    x = BatchNormalization()(x)
    
    x = tf.keras.layers.SeparableConv1D(16, 7, strides=4, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    
    x = tf.keras.layers.SeparableConv1D(32, 7, strides=2, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    
    x = tf.keras.layers.SeparableConv1D(64, 7, strides=4, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    
    x = tf.keras.layers.SeparableConv1D(128, 7, strides=2, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    # SE block 적용하여 채널 중요도 보정
    x = se_block(x, reduction=4)
    x = GlobalAveragePooling1D()(x)
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
# PPG 단일 모달 모델 생성 개선:
# - 입력 직후 BatchNormalization을 적용하여 raw 신호의 분포를 정규화
# =============================================================================
def create_ppg_model(input_shape, dropout_rate=DROPOUT_RATE, num_classes=NUM_CLASSES):
    ppg_input = Input(shape=input_shape, name='ppg_input')
    norm = BatchNormalization()(ppg_input)
    features = create_ppg_feature_extractor(input_shape, num_branches=input_shape[-1])(norm)
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
                                          result_dir_base="/home/bcml1/sigenv/_4주차_ppg/jh_result5_inter_seg10s"):
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
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
    
    print("=== Sample-level Evaluation ===")
    print(overall_report)
    
    # 각 trial별로 개별적인 classification report 생성 (파일명에 trial 번호 사용)
    unique_trials = np.unique(test_trial_ids)
    for trial in unique_trials:
        trial_indices = np.where(test_trial_ids == trial)[0]
        if len(trial_indices) == 0:
            continue
        trial_y_true = np.argmax(test_y[trial_indices], axis=1)
        trial_y_pred = np.argmax(y_pred[trial_indices], axis=1)
        trial_report = classification_report(trial_y_true, trial_y_pred, digits=4)
        trial_cm = confusion_matrix(trial_y_true, trial_y_pred)
        report_file = os.path.join(result_dir, f"classification_report_trial_{trial:02d}.txt")
        cm_file = os.path.join(result_dir, f"confusion_matrix_trial_{trial:02d}.txt")
        with open(report_file, "w") as f:
            f.write(trial_report)
        with open(cm_file, "w") as f:
            f.write(np.array2string(trial_cm))
        print(f"=== Trial {trial:02d} Evaluation ===")
        print(trial_report)
    
    return model, history

# =============================================================================
# Main 함수: PPG 데이터만 로딩 후, leave-one-subject-out 방식으로 단일 모달 학습 및 평가
# =============================================================================
def main():
    ppg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_1s"
    
    # PPG 데이터 로딩 (동일한 WINDOW_SIZE, SLIDING_STEP 적용)
    ppg_subject_data = load_ppg_data(ppg_data_dir, ppg_segment_length=PPG_SEGMENT_LENGTH, 
                                     window_size=WINDOW_SIZE, step=SLIDING_STEP)
    
    # ppg_subject_data의 각 key는 subject_id (예: "s01", "s02", ...)
    subject_data = {}
    for subject in sorted(ppg_subject_data.keys()):
        print(f"--- {subject} 데이터 로딩 시작 ---")
        ppg_data = ppg_subject_data[subject]
        subject_data[subject] = {
            'ppg_X': ppg_data['X'],
            'y': to_categorical(ppg_data['y'], num_classes=NUM_CLASSES),
            'trial_ids': ppg_data['trial_ids']  # load_ppg_data에서 추출한 trial id 사용
        }
        print(f"{subject} - PPG data shape: {ppg_data['X'].shape}, y shape: {ppg_data['y'].shape}, "
              f"trial_ids shape: {ppg_data['trial_ids'].shape}")
    
    result_dir_base = "/home/bcml1/sigenv/_4주차_ppg/jh_result5_inter_seg10s"
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
