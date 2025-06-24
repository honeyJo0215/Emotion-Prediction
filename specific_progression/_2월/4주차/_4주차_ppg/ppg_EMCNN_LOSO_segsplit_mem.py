import os
import os
import glob
import numpy as np
import tensorflow as tf
import scipy.signal as signal
from tensorflow.keras.layers import Input, Conv1D, DepthwiseConv1D, Dense, GlobalAveragePooling1D, concatenate, BatchNormalization, Activation, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical

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

limit_gpu_memory(7000)

# =============================================================================
# EMCNN 모델 구성 (입력 shape: (1280, 5))
# =============================================================================
input_shape = (1280, 5)
num_classes = 4  # 예: HVHA, HVLA, LVLA, LVHA

# === MobileNet 기반 Feature Extraction CNN ===
def emcnn_branch(input_layer):
    """ EMCNN의 각 branch에 해당하는 CNN 블록 (MobileNet 기반) """
    x = Conv1D(8, kernel_size=7, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(input_layer)
    
    # Depthwise Separable Convolution 적용
    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(16, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(32, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    # Global Average Pooling 적용
    x = GlobalAveragePooling1D()(x)
    return x

# =============================================================================
# EMCNN 전체 모델 (입력 shape: (1280, 5))
# =============================================================================
def build_emcnn():
    inputs = Input(shape=input_shape)  # 각 sample의 shape: (1280, 5)

    # Lambda Layer를 사용하여 각 채널(branch)를 추출 (axis=2에서 채널 추출)
    branch1 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 0], axis=-1))(inputs))
    branch2 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 1], axis=-1))(inputs))
    branch3 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 2], axis=-1))(inputs))
    branch4 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 3], axis=-1))(inputs))
    branch5 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 4], axis=-1))(inputs))
    
    merged = concatenate([branch1, branch2, branch3, branch4, branch5])
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(merged)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# =============================================================================
# 슬라이딩 윈도우 함수
# =============================================================================
def sliding_window_ppg(continuous_data, window_size, step):
    win_length = window_size * 128
    step_length = step * 128
    windows = [continuous_data[i:i + win_length] for i in range(0, len(continuous_data) - win_length + 1, step_length)]
    return np.array(windows, dtype=np.float32)

# =============================================================================
# PPG 데이터 파일 경로와 라벨 정보를 subject 단위로 저장하는 함수 (trial 단위)
# 각 파일은 (60, 5, 128) shape이며, 60개의 segment를 (60,128,5)로 변환 후 이어붙여 (7680,5)가 됨.
# 슬라이딩 윈도우를 적용하면 각 파일에서 51개의 window(=segment)가 생성됨.
# =============================================================================
def load_ppg_data_with_sliding_window(ppg_data_dir, window_size=10, step=1):
    subject_data = {}
    file_paths = glob.glob(os.path.join(ppg_data_dir, "*.npy"))
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

        data = np.load(file_path)
        # (60, 5, 128) → (60, 128, 5)
        data = np.transpose(data, (0, 2, 1))
        # 연속 신호 (7680, 5)
        continuous_data = np.concatenate(data, axis=0)
        # 슬라이딩 윈도우 적용: 결과 shape=(51, 1280, 5)
        windows = sliding_window_ppg(continuous_data, window_size, step)
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(windows)
        subject_data[subject_id]['y'].append(label)
    for subject in subject_data:
        num_trials = len(subject_data[subject]['X'])
        print(f"{subject} - Trial 수: {num_trials}")
    return subject_data

# =============================================================================
# PPG 데이터 증강 함수들 (동일 코드 유지)
# =============================================================================
def augment_ppg(ppg, noise_std=0.01, shift_max=5, scale_range=(0.9, 1.1), mask_freq=10, mask_factor=0.2, peak_shift=3):
    augmented_ppg = np.zeros_like(ppg)
    for ch in range(ppg.shape[1]):
        ppg_ch = ppg[:, ch]
        ppg_noisy = ppg_ch + np.random.normal(0, noise_std, ppg_ch.shape)
        shift = np.random.randint(-shift_max, shift_max)
        ppg_shifted = np.roll(ppg_noisy, shift, axis=0)
        scale_factor = np.random.uniform(*scale_range)
        ppg_scaled = ppg_shifted * scale_factor
        fft_signal = np.fft.fft(ppg_scaled)
        fft_signal[:mask_freq] *= (1 - mask_factor)
        ppg_freq = np.real(np.fft.ifft(fft_signal))
        peaks, _ = signal.find_peaks(ppg_freq, distance=30)
        if len(peaks) > 0:
            shift_values = np.random.randint(-peak_shift, peak_shift, size=len(peaks))
            for i, s in zip(peaks, shift_values):
                if 0 <= i + s < len(ppg_freq):
                    ppg_freq[i] = ppg_freq[i + s]
        augmented_ppg[:, ch] = ppg_freq
    return augmented_ppg

def balance_ppg_dataset(data_X, _, y, noise_std_ppg=0.01):
    y_labels = np.argmax(y, axis=1)
    classes = np.unique(y_labels)
    counts = {cls: np.sum(y_labels == cls) for cls in classes}
    max_count = max(counts.values())
    X_list = [data_X]
    y_list = [y]
    for cls in classes:
        indices = np.where(y_labels == cls)[0]
        num_needed = max_count - len(indices)
        for _ in range(num_needed):
            idx = np.random.choice(indices)
            sample = data_X[idx]
            aug_sample = augment_ppg(sample, noise_std=noise_std_ppg)
            X_list.append(aug_sample[np.newaxis, ...])
            y_list.append(y[idx][np.newaxis, ...])
    X_augmented = np.concatenate(X_list, axis=0)
    y_augmented = np.concatenate(y_list, axis=0)
    idxs = np.arange(len(y_augmented))
    np.random.shuffle(idxs)
    return X_augmented[idxs], X_augmented[idxs], y_augmented[idxs]

# =============================================================================
# 학습 곡선(accuracy, loss) 플롯 저장 함수
# =============================================================================
def plot_training_curves(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# inter-subject 평가: 각 subject에 대해 trial 데이터를 51 segment 단위로 사용하여
# tf.data.Dataset으로 로드 후 학습 및 평가하는 함수
# =============================================================================
def train_subject_leave_one_out(subject_data, epochs=300, batch_size=32):
    parent_dir = "/home/bcml1/sigenv/_4주차_ppg/LOSO_result_segsplit_nonorm"
    os.makedirs(parent_dir, exist_ok=True)
    subject_ids = sorted(subject_data.keys())
    
    for test_subj in subject_ids:
        print(f"\nLOSO - Test Subject: {test_subj}")
        # 테스트 subject 제외한 나머지의 trial 데이터를 수집 (trial: shape=(51, 1280, 5))
        trial_list = []
        label_list = []
        for subj in subject_ids:
            if subj == test_subj:
                continue
            for trial, label in zip(subject_data[subj]['X'], subject_data[subj]['y']):
                trial_list.append(trial)
                label_list.append(label)
        
        # trial 단위로 80:20 학습/검증 분할 (stratify 적용)
        trials_train, trials_val, labels_train, labels_val = train_test_split(
            trial_list, label_list, test_size=0.2, random_state=42, stratify=label_list
        )
        
        # 각 trial은 (51, 1280, 5) shape이므로, 모든 trial의 윈도우들을 하나로 결합
        X_train = np.concatenate(trials_train, axis=0)  # shape: (total_train_windows, 1280, 5)
        y_train = np.concatenate([np.full((trial.shape[0],), label, dtype=np.int32)
                                  for trial, label in zip(trials_train, labels_train)], axis=0)
        X_val = np.concatenate(trials_val, axis=0)
        y_val = np.concatenate([np.full((trial.shape[0],), label, dtype=np.int32)
                                for trial, label in zip(trials_val, labels_val)], axis=0)
        
        print("X_train shape:", X_train.shape)  # 예: (train_windows, 1280, 5)
        
        # one-hot 인코딩
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)
        
        # 증강 적용 (PPG 데이터 증강)
        X_train, _, y_train = balance_ppg_dataset(X_train, X_train, y_train, noise_std_ppg=0.01)
        X_val, _, y_val = balance_ppg_dataset(X_val, X_val, y_val, noise_std_ppg=0.01)
        
        # 테스트 subject의 데이터 (trial 단위, 나중에 trial별 평가에 사용)
        X_test = np.concatenate(subject_data[test_subj]['X'], axis=0)
        y_test_int = np.array(subject_data[test_subj]['y'])
        y_test = to_categorical(y_test_int, num_classes=num_classes)
        
        # tf.data.Dataset으로 변환 (메모리 관리 및 전처리 최적화 적용)
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # 모델 생성 및 컴파일
        model = build_emcnn()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)
        
        result_dir = os.path.join(parent_dir, f"test_{test_subj}")
        os.makedirs(result_dir, exist_ok=True)
        
        model.save(os.path.join(result_dir, f"{test_subj}_model.keras"))
        np.save(os.path.join(result_dir, f"{test_subj}_history.npy"), history.history)
        plot_training_curves(history, os.path.join(result_dir, f"{test_subj}_training_curves.png"))
        
        # 1) 전체 테스트 subject의 윈도우 평가
        y_pred_test = model.predict(test_ds)
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred_test, axis=1)
        overall_report = classification_report(y_test_labels, y_pred_labels, digits=4)
        overall_cm = confusion_matrix(y_test_labels, y_pred_labels)
        with open(os.path.join(result_dir, "classification_report_overall.txt"), "w") as f:
            f.write(overall_report)
        with open(os.path.join(result_dir, "confusion_matrix_overall.txt"), "w") as f:
            f.write(np.array2string(overall_cm))
        
        # 2) Trial 단위 평가 (각 trial 별로 평가)
        for idx, trial in enumerate(subject_data[test_subj]['X']):
            trial_preds = model.predict(trial)
            y_true_trial = np.full((trial.shape[0],), subject_data[test_subj]['y'][idx], dtype=np.int32)
            y_true_trial = to_categorical(y_true_trial, num_classes=num_classes)
            y_true_labels = np.argmax(y_true_trial, axis=1)
            y_pred_trial = np.argmax(trial_preds, axis=1)
            report_trial = classification_report(y_true_labels, y_pred_trial, digits=4)
            cm_trial = confusion_matrix(y_true_labels, y_pred_trial)
            with open(os.path.join(result_dir, f"classification_report_trial_{idx:02d}.txt"), "w") as f:
                f.write(report_trial)
            with open(os.path.join(result_dir, f"confusion_matrix_trial_{idx:02d}.txt"), "w") as f:
                f.write(np.array2string(cm_trial))
            print(f"Test subject {test_subj} - Trial {idx:02d} report:")
            print(report_trial)
    
if __name__ == "__main__":
    # data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'
    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s_nonorm'
    # window_size=10초, step=1초로 슬라이딩 윈도우 적용 → 각 trial의 shape: (num_windows, 1280, 5)
    subject_data = load_ppg_data_with_sliding_window(data_dir, window_size=10, step=1)
    train_subject_leave_one_out(subject_data, epochs=300, batch_size=32)
    