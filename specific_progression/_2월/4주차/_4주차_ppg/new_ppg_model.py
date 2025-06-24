import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Bidirectional, concatenate
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import BatchNormalization, Dropout
from numpy.lib.stride_tricks import sliding_window_view

# ==============================
# 1. 데이터 로드 및 변환 (10초 단위 + 1초 오버랩)
# ==============================
DATA_DIR = "/home/bcml1/2025_EMOTION/DEAP_PPG_1s"
INTRA_RESULTS_DIR = "./new_PPG_intra_tf"
INTER_RESULTS_DIR = "./new_PPG_inter_tf"

LABEL_MAP = {"0": 0, "1": 1, "2": 2, "3": 3}  # excited(0), relaxed(1), stressed(2), bored(3)

# 폴더 생성
os.makedirs(INTRA_RESULTS_DIR, exist_ok=True)
os.makedirs(INTER_RESULTS_DIR, exist_ok=True)

def load_data(data_dir, window_size=10, overlap_size=1):
    """
    PPG 데이터를 window_size 초 단위로 분할하고 overlap_size 초 단위로 오버랩하여 변환
    :param data_dir: 데이터가 저장된 디렉토리
    :param window_size: n초 단위 (기본: 10초)
    :param overlap_size: m초 오버랩 (기본: 1초, 0.5초도 가능)
    :return: 변환된 데이터 (X, y, subjects, trials)
    """
    file_list = glob.glob(os.path.join(data_dir, "*.npy"))
    data_list, labels, subjects, trials = [], [], [], []
    
    samples_per_sec = 128
    window_size_samples = int(window_size * samples_per_sec)  # 예: 10초 -> 1280 샘플
    step_size = window_size_samples - int(overlap_size * samples_per_sec)
    
    for file in file_list:
        file_name = os.path.basename(file)
        parts = file_name.split("_")
        subject_id = parts[0]  # 예: "sXX"
        trial_id = parts[2]    # 예: "trial_XX"
        label_str = parts[-1].replace(".npy", "")
        # 감정 라벨 매핑
        LABEL_MAP = {"0": 0, "1": 1, "2": 2, "3": 3}
        label = LABEL_MAP[label_str]
        
        npy_data = np.load(file)  # 원본 shape: (60, 5, 128)
        if npy_data.shape != (60, 5, 128):
            print(f"❌ 데이터 형식 오류: {file} - Shape {npy_data.shape}")
            continue
        
        # 각 초별 데이터를 하나의 연속 신호로 병합 (복사 없이 뷰 생성)
        continuous_data = np.transpose(npy_data, (0, 2, 1)).reshape(-1, 5)  # shape: (7680, 5)
        
        # sliding_window_view를 사용하여 (window_size_samples, 5) 크기의 뷰를 생성 (복사하지 않음)
        windows_view = sliding_window_view(continuous_data, window_shape=(window_size_samples, 5))
        # windows_view의 첫 번째 차원은 모든 가능한 창이므로, step_size 간격으로 선택합니다.
        windowed_data = windows_view[::step_size, 0, :, :]  # shape: (N, window_size_samples, 5)
        
        data_list.append(windowed_data)
        labels.extend([label] * windowed_data.shape[0])
        subjects.extend([subject_id] * windowed_data.shape[0])
        trials.extend([trial_id] * windowed_data.shape[0])
    
    if data_list:
        X = np.concatenate(data_list, axis=0)
        y = np.array(labels)
        subjects = np.array(subjects)
        trials = np.array(trials)
        return X, y, subjects, trials
    else:
        print("⚠ No valid data found!")
        return np.array([]), np.array([]), np.array([]), np.array([])

# ==============================
# 2. TensorFlow Dataset 정의
# ==============================
def create_tf_dataset(X, y, batch_size=16, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(len(X))
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# ==============================
# 3. CNN-LSTM 모델 정의
# ==============================
# def build_cnn_lstm_model():
#     input_layer = Input(shape=(5, 1280))  # (channels, time_steps)

#     # CNN Layers (Multi-scale Feature Extraction)
#     conv1 = Conv1D(32, kernel_size=3, padding="same", activation="relu")(input_layer)
#     conv1 = BatchNormalization()(conv1)
#     conv2 = Conv1D(32, kernel_size=5, padding="same", activation="relu")(input_layer)
#     conv2 = BatchNormalization()(conv2)
#     conv3 = Conv1D(32, kernel_size=7, padding="same", activation="relu")(input_layer)
#     conv3 = BatchNormalization()(conv3)

#     # Max Pooling (Size Reduction)
#     pool1 = MaxPooling1D(pool_size=4)(conv1)  # (batch, 320, 32)
#     pool2 = MaxPooling1D(pool_size=4)(conv2)  # (batch, 320, 32)
#     pool3 = MaxPooling1D(pool_size=4)(conv3)  # (batch, 320, 32)

#     # Multi-scale Feature Fusion
#     merged = concatenate([pool1, pool2, pool3])  # (batch_size, 320, 96)

#     # LSTM Layer
#     lstm = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3))(merged)

#     # Fully Connected Layer
#     dense = Dense(64, activation="relu")(lstm)
#     dense = Dropout(0.3)(dense)
#     output_layer = Dense(4, activation="softmax")(dense)  # 4-class classification (excited, relaxed, stressed, bored)

#     # Model Compilation
#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
#     return model
def build_flexible_cnn_lstm_model(time_steps):
    input_layer = Input(shape=(5, time_steps))  # (channels, time_steps)

    # CNN Layers (Multi-scale Feature Extraction)
    conv1 = Conv1D(32, kernel_size=3, padding="same", activation="relu")(input_layer)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(32, kernel_size=5, padding="same", activation="relu")(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(32, kernel_size=7, padding="same", activation="relu")(pool2)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    # Multi-scale Feature Fusion
    merged = concatenate([pool1, pool2, pool3])  # (batch_size, reduced_T, 96)

    # LSTM Layer (Sequence Learning)
    lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(merged)
    lstm = LSTM(64, return_sequences=False)(lstm)  # 추가 레이어

    # Fully Connected Layer
    dense = Dense(64, activation="relu")(lstm)
    dense = Dropout(0.3)(dense)
    output_layer = Dense(4, activation="softmax")(dense)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model

# ==============================
# 4. Training Curve 및 Confusion Matrix 저장 함수
# ==============================
def save_training_curves(history, results_dir, fold):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"Training Curve - {fold}")
    plt.savefig(os.path.join(results_dir, f"training_curve_{fold}.png"))
    plt.close()

def save_confusion_matrix(y_true, y_pred, results_dir, fold):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_MAP.keys(), yticklabels=LABEL_MAP.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {fold}")
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{fold}.png"))
    plt.close()
    
# ==============================
# 5. Inter-subject Cross-validation (Leave-One-Subject-Out)
# ==============================
def inter_subject_train(window_size=10, overlap_size=1):
    unique_subjects = np.unique(subjects)
    
    for test_subject in unique_subjects:
        print(f"\n===== Leave-One-Subject-Out: Testing on {test_subject} ({window_size}s window) =====")
        subject_dir = os.path.join(INTER_RESULTS_DIR, f"subject_{test_subject}")
        os.makedirs(subject_dir, exist_ok=True)

        # ✅ 훈련 및 테스트 데이터 분할
        train_indices = [i for i, s in enumerate(subjects) if s != test_subject]
        test_indices = [i for i, s in enumerate(subjects) if s == test_subject]

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # ✅ 훈련/검증 데이터셋 분할 (훈련:검증 = 9:1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        train_dataset = create_tf_dataset(X_train, y_train, batch_size=32)
        val_dataset = create_tf_dataset(X_val, y_val, shuffle=False, batch_size=32)
        test_dataset = create_tf_dataset(X_test, y_test, shuffle=False, batch_size=32)

        # ✅ 유연한 CNN-LSTM 모델 생성 (time_steps 자동 설정)
        time_steps = window_size * 128  # 샘플 개수 (128Hz)
        print(f"Time steps for model ({window_size}s window):", time_steps)
        model = build_flexible_cnn_lstm_model(time_steps)
        
        # ✅ 모델 학습
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=10, verbose=2)

        # ✅ 평가
        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, digits=4)
        print(f"\n===== Classification Report for {test_subject} ({window_size}s window) =====\n{report}")

        # ✅ 결과 저장
        save_training_curves(history, subject_dir, str(test_subject))
        save_confusion_matrix(y_test, y_pred, subject_dir, str(test_subject))

        with open(os.path.join(subject_dir, "classification_report.txt"), "w") as f:
            f.write(report + "\n")

# ==============================
# 6. Intra-subject Cross-validation (5-fold)
# ==============================
def intra_subject_train(window_size=10, overlap_size=1):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n===== Fold {fold}/5 ({window_size}s window) =====")
        fold_dir = os.path.join(INTRA_RESULTS_DIR, f"kfold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # ✅ 훈련/검증 데이터셋 분할 (훈련:검증 = 9:1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        train_dataset = create_tf_dataset(X_train, y_train, batch_size=32)
        val_dataset = create_tf_dataset(X_val, y_val, shuffle=False, batch_size=32)
        test_dataset = create_tf_dataset(X_test, y_test, shuffle=False, batch_size=32)

        # ✅ 유연한 CNN-LSTM 모델 생성 (time_steps 자동 설정)
        time_steps = window_size * 128  # 샘플 개수 (128Hz)
        print(f"Time steps for model ({window_size}s window):", time_steps)
        model = build_flexible_cnn_lstm_model(time_steps)

        # ✅ 모델 학습
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=10, verbose=2)

        # ✅ 평가
        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, digits=4)
        print(f"\n===== Fold {fold} Classification Report ({window_size}s window) =====\n{report}")

        # ✅ 결과 저장
        save_training_curves(history, fold_dir, str(fold))
        save_confusion_matrix(y_test, y_pred, fold_dir, str(fold))

        with open(os.path.join(fold_dir, "classification_report.txt"), "w") as f:
            f.write(report + "\n")

# 실행
X, y, subjects, trials = load_data(DATA_DIR, window_size=10, overlap_size=0.5)

# ✅ 데이터 로딩이 실패했을 경우 프로그램 종료
if X.shape[0] == 0:
    print("🚨 Error: No valid data loaded! Check data directory and preprocessing.")
    exit(1)

# ✅ 데이터 크기 확인
print("X shape:", X.shape)  # 예상: (샘플 수, 5, 1280)
print("y shape:", y.shape)  # 예상: (샘플 수,)
print("subjects shape:", subjects.shape)  # 예상: (샘플 수,)
print("trials shape:", trials.shape)  # 예상: (샘플 수,)

# 학습 시작
inter_subject_train()
intra_subject_train()
