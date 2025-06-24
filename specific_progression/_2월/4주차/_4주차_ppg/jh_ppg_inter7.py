import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import mode
from scipy.signal import butter, filtfilt, detrend
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Input, Dense, BatchNormalization, Dropout, 
                                     Conv2D, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===== GPU 메모리 제한 =====
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

# ===== 전역 하이퍼파라미터 =====
WINDOW_SIZE = 10
SLIDING_STEP = 1
PPG_SEGMENT_LENGTH = 128
EPOCHS = 150
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
NUM_CLASSES = 4
BINS = 64  # 2D 맵 해상도

# ===== 필터, 전처리 함수 =====
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def preprocess_ppg_trial(trial_data, fs=128):
    processed_segments = []
    for segment in trial_data:  # segment shape: (128, 5)
        detrended = detrend(segment, axis=0)
        filtered = np.zeros_like(detrended)
        for ch in range(detrended.shape[1]):
            filtered[:, ch] = bandpass_filter(detrended[:, ch], lowcut=0.5, highcut=8.0, fs=fs, order=4)
        smoothed = np.zeros_like(filtered)
        for ch in range(filtered.shape[1]):
            smoothed[:, ch] = moving_average(filtered[:, ch], window_size=3)
        normalized = (smoothed - np.mean(smoothed, axis=0)) / (np.std(smoothed, axis=0) + 1e-8)
        processed_segments.append(normalized)
    return np.array(processed_segments)  # shape: (num_segments, 128, 5)

def subject_wise_normalize(X):
    # X shape: (num_samples, 1280, 5) - (window_size=10 기준)
    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True)
    return (X - mean) / (std + 1e-8)

# ===== 카오틱 맵 2D 이미지 함수 =====
def chaotic_map_2d_image(signal_1d, bins=64, downsample_factor=10):
    ds_signal = signal_1d[::downsample_factor]
    N = len(ds_signal)
    k = np.arange(1, N + 1)
    a = np.clip(ds_signal, -1, 1)
    theta = np.arccos(a)
    rho = np.cos(k * theta)
    X = rho * np.cos(a)
    Y = rho * np.sin(a)
    # 클리핑 (-1,1)
    X = np.clip(X, -1, 1)
    Y = np.clip(Y, -1, 1)
    # [-1,1] -> [0, bins-1]
    x_scaled = (X + 1) / 2 * (bins - 1)
    y_scaled = (Y + 1) / 2 * (bins - 1)
    x_idx = np.clip(x_scaled.round().astype(int), 0, bins - 1)
    y_idx = np.clip(y_scaled.round().astype(int), 0, bins - 1)
    hist_2d = np.zeros((bins, bins), dtype=np.float32)
    for i in range(len(x_idx)):
        hist_2d[y_idx[i], x_idx[i]] += 1.0
    if hist_2d.max() > 0:
        hist_2d /= hist_2d.max()
    return hist_2d

# ===== PPG 데이터 로딩 (sliding window + 채널별 2D chaotic map) =====
def load_ppg_data(data_dir='/home/bcml1/2025_EMOTION/DEAP_PPG_1s',
                  ppg_segment_length=PPG_SEGMENT_LENGTH,
                  window_size=WINDOW_SIZE,
                  step=SLIDING_STEP,
                  bins=BINS):
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
        subject_id = match.group(1)
        trial_id = int(match.group(2))
        label = int(match.group(3))
        
        # data shape: (60, 5, 128) => (60, 128, 5)
        data = np.load(file_path)
        data = np.transpose(data, (0, 2, 1))
        data = data[:, :ppg_segment_length, :]
        data = preprocess_ppg_trial(data, fs=128)  # (num_segments, 128, 5)
        
        num_segments = data.shape[0]
        windowed_samples = []
        trial_ids_list = []
        
        # Sliding window
        for start in range(0, num_segments - window_size + 1, step):
            window = data[start:start + window_size]  # shape (window_size, 128, 5)
            combined = np.concatenate(window, axis=0)  # (128*window_size, 5) = (1280, 5)
            windowed_samples.append(combined)
            trial_ids_list.append(trial_id)
        
        if not windowed_samples:
            print(f"유효한 sliding window가 없습니다: {file_path}")
            continue
        
        windowed_samples = np.array(windowed_samples)  # (num_samples, 1280, 5)
        labels_array = np.full((windowed_samples.shape[0],), label, dtype=np.int32)
        
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': [], 'trial_ids': []}
        subject_data[subject_id]['X'].append(windowed_samples)
        subject_data[subject_id]['y'].append(labels_array)
        subject_data[subject_id]['trial_ids'].append(np.array(trial_ids_list))
    
    # 각 subject별로 데이터 합치기
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)   # (N, 1280, 5)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)   # (N,)
        subject_data[subject]['trial_ids'] = np.concatenate(subject_data[subject]['trial_ids'], axis=0)  # (N,)
        
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
                # 간단한 가우시안 노이즈
                noise = np.random.normal(0.0, noise_std, X_samples.shape)
                X_aug = X_samples + noise
                X_aug_list.append(X_aug)
                y_aug_list.append(np.full(diff, cls, dtype=y_data.dtype))
                trial_ids_list.append(subject_data[subject]['trial_ids'][sampled_indices])
        
        subject_data[subject]['X'] = np.concatenate(X_aug_list, axis=0)
        subject_data[subject]['y'] = np.concatenate(y_aug_list, axis=0)
        subject_data[subject]['trial_ids'] = np.concatenate(trial_ids_list, axis=0)
        
        # subject-wise 정규화
        subject_data[subject]['X'] = subject_wise_normalize(subject_data[subject]['X'])
        
        # === 채널별 2D 카오틱 맵 생성 & 스택 ===
        # shape: (N, 1280, 5)
        X = subject_data[subject]['X']
        chaotic_maps = []
        for i in range(X.shape[0]):  # 모든 sample에 대해
            # sample shape: (1280, 5)
            sample_5ch = X[i]  # (1280, 5)
            
            channel_maps = []
            for ch in range(sample_5ch.shape[1]):
                # 한 채널 (1280,)을 2D 카오틱 맵으로 변환
                map_2d = chaotic_map_2d_image(sample_5ch[:, ch], bins=bins, downsample_factor=10)
                channel_maps.append(map_2d[..., np.newaxis])  # (bins, bins, 1)
            
            # 5개 채널 스택 -> (bins, bins, 5)
            channel_maps = np.concatenate(channel_maps, axis=-1)
            chaotic_maps.append(channel_maps)
        
        chaotic_maps = np.array(chaotic_maps, dtype=np.float32)  # (N, bins, bins, 5)
        subject_data[subject]['chaotic_map'] = chaotic_maps
        
        print(f"{subject} after augmentation & chaotic 2D map creation: "
              f"X shape: {subject_data[subject]['X'].shape}, "
              f"chaotic_map shape: {chaotic_maps.shape}, "
              f"y shape: {subject_data[subject]['y'].shape}, "
              f"trial_ids shape: {subject_data[subject]['trial_ids'].shape}")
    
    return subject_data

# ===== 2D CNN 모델 (입력: (bins, bins, 5)) =====
def create_chaotic_model_2d(input_shape, dropout_rate=DROPOUT_RATE, num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape)  # (bins, bins, 5)
    x = BatchNormalization()(inputs)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

# ===== 데이터 증강 (chaotic map) =====
def balance_dataset(features, y, noise_std=0.01):
    # features: (N, bins, bins, 5)
    y_labels = np.argmax(y, axis=1)
    classes = np.unique(y_labels)
    counts = {cls: np.sum(y_labels == cls) for cls in classes}
    max_count = max(counts.values())
    
    X_list = [features]
    y_list = [y]
    for cls in classes:
        indices = np.where(y_labels == cls)[0]
        num_needed = max_count - len(indices)
        for _ in range(num_needed):
            idx = np.random.choice(indices)
            sample = features[idx]
            # 2D 맵 전체에 노이즈 추가
            X_aug = sample + np.random.normal(0.0, noise_std, sample.shape)
            X_list.append(X_aug[np.newaxis, ...])
            y_list.append(y[idx][np.newaxis, ...])
    
    X_augmented = np.concatenate(X_list, axis=0)
    y_augmented = np.concatenate(y_list, axis=0)
    idxs = np.arange(len(y_augmented))
    np.random.shuffle(idxs)
    return X_augmented[idxs], y_augmented[idxs]

# ===== 학습 기록 저장 =====
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
    os.makedirs(fold_result_dir, exist_ok=True)
    plt.savefig(os.path.join(fold_result_dir, "training_history.png"))
    plt.close()

# ===== Leave-One-Subject-Out 학습 =====
def train_chaotic_model_2d_leave_one_subject_out(test_subject_id, train_features, train_y,
                                                 test_features, test_y, test_trial_ids,
                                                 num_classes=NUM_CLASSES, epochs=EPOCHS,
                                                 batch_size=BATCH_SIZE,
                                                 result_dir_base="/home/bcml1/sigenv/_4주차_ppg/jh_result_chaotic2d"):
    result_dir = os.path.join(result_dir_base, f"test_{test_subject_id}")
    os.makedirs(result_dir, exist_ok=True)
    
    X_train, X_val, y_train, y_val = train_test_split(
        train_features, train_y, test_size=0.2, random_state=42, stratify=np.argmax(train_y, axis=1)
    )
    
    # 클래스 불균형 보정
    X_train, y_train = balance_dataset(X_train, y_train, noise_std=0.01)
    X_val, y_val = balance_dataset(X_val, y_val, noise_std=0.01)
    
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (bins, bins, 5)
    model = create_chaotic_model_2d(input_shape, dropout_rate=DROPOUT_RATE, num_classes=num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint_path = os.path.join(result_dir, "best_model.keras")
    checkpoint_cb = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[checkpoint_cb, earlystop_cb], verbose=1)
    
    save_training_history(history, result_dir, test_subject_id, fold_no=0)
    
    model.load_weights(checkpoint_path)
    y_pred = model.predict(test_features)
    
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
    
    # trial별로 리포트
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

# ===== Main 함수 =====
def main():
    ppg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_1s"
    ppg_subject_data = load_ppg_data(ppg_data_dir,
                                     ppg_segment_length=PPG_SEGMENT_LENGTH,
                                     window_size=WINDOW_SIZE,
                                     step=SLIDING_STEP,
                                     bins=BINS)
    
    subject_data = {}
    for subject in sorted(ppg_subject_data.keys()):
        print(f"--- {subject} 데이터 로딩 시작 ---")
        ppg_data = ppg_subject_data[subject]
        subject_data[subject] = {
            'chaotic_map': ppg_data['chaotic_map'],  # shape: (N, bins, bins, 5)
            'y': to_categorical(ppg_data['y'], num_classes=NUM_CLASSES),
            'trial_ids': ppg_data['trial_ids']
        }
        print(f"{subject} - chaotic_map shape: {ppg_data['chaotic_map'].shape}, "
              f"y shape: {ppg_data['y'].shape}, trial_ids shape: {ppg_data['trial_ids'].shape}")
    
    result_dir_base = "/home/bcml1/sigenv/_4주차_ppg/jh_result_chaotic2d"
    os.makedirs(result_dir_base, exist_ok=True)
    
    # Leave-One-Subject-Out
    for test_subject in subject_data.keys():
        print(f"--- Leave-One-Subject-Out: Test subject {test_subject} ---")
        train_feat_list, train_y_list = [], []
        for subject in subject_data:
            if subject == test_subject:
                continue
            train_feat_list.append(subject_data[subject]['chaotic_map'])
            train_y_list.append(subject_data[subject]['y'])
        train_feat = np.concatenate(train_feat_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        
        test_feat = subject_data[test_subject]['chaotic_map']
        test_y = subject_data[test_subject]['y']
        test_trial_ids = subject_data[test_subject]['trial_ids']
        
        train_chaotic_model_2d_leave_one_subject_out(
            test_subject,
            train_feat, train_y,
            test_feat, test_y,
            test_trial_ids,
            num_classes=NUM_CLASSES,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            result_dir_base=result_dir_base
        )

if __name__ == "__main__":
    main()
