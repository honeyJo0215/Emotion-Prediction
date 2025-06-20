import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 데이터 로드 함수

def load_segmented_data(data_dir):
    """
    세분화된 데이터를 로드합니다.

    Args:
        data_dir (str): EEG 데이터가 저장된 디렉토리 경로.

    Returns:
        np.array, np.array, np.array: 특징(feature) 데이터, 라벨(label) 데이터, 참가자 데이터.
    """
    X = []
    y = []
    participants = []

    for file in os.listdir(data_dir):
        if file.endswith("_FB.npy"):
            file_path = os.path.join(data_dir, file)
            data = np.load(file_path, allow_pickle=True)  # (samples, timesteps, channels)

            label = 1 if "positive" in file else 0
            participant = file.split('_')[0]  # e.g., s01

            for sample in data:
                X.append(sample)
                y.append(label)
                participants.append(participant)

    X = np.array(X)
    y = np.array(y)
    participants = np.array(participants)

    return X, y, participants

# Sequential API 모델 정의

def build_model(input_shape):
    """
    Sequential API 모델 정의.

    Args:
        input_shape (tuple): 입력 데이터 형태.

    Returns:
        model (tf.keras.Model): 컴파일된 Conv1D 모델.
    """
    if len(input_shape) != 2:
        raise ValueError(f"Expected input shape to have 2 dimensions (timesteps, channels), got {input_shape}.")

    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 데이터 분리 및 학습 함수

def train_intra_subject(data_dir, result_dir, model_dir):
    """
    Sequential API 모델을 사용하여 raw 데이터를 intra-subject 방식으로 학습 및 평가.

    Args:
        data_dir (str): EEG 데이터가 저장된 디렉토리 경로.
        result_dir (str): 평가 결과를 저장할 디렉토리 경로.
        model_dir (str): 모델을 저장할 디렉토리 경로.
    """
    participants = np.unique([file.split('_')[0] for file in os.listdir(data_dir) if file.endswith("_FB.npy")])

    for participant in participants:
        print(f"Processing participant {participant}...")

        # 데이터 로드
        X, y, participants_data = load_segmented_data(data_dir)

        # Participant 데이터 필터링
        participant_mask = participants_data == participant
        X_participant = X[participant_mask]
        y_participant = y[participant_mask]

        # Train/Validation/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_participant, y_participant, test_size=0.2, random_state=42, stratify=y_participant
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        # Normalize data
        X_train = X_train / (np.max(np.abs(X_train), axis=(1, 2), keepdims=True) + 1e-8)
        X_val = X_val / (np.max(np.abs(X_val), axis=(1, 2), keepdims=True) + 1e-8)
        X_test = X_test / (np.max(np.abs(X_test), axis=(1, 2), keepdims=True) + 1e-8)

        # # Reshape data for Conv1D
        # X_train = np.expand_dims(X_train, axis=-1)
        # X_val = np.expand_dims(X_val, axis=-1)
        # X_test = np.expand_dims(X_test, axis=-1)

        model = build_model(X_train.shape[1:])

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30, batch_size=32, verbose=1
        )

        # 모델 평가
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        report = classification_report(y_test, y_pred, target_names=["negative", "positive"], output_dict=False)
        print(report)

        # 모델 저장
        model_save_path = os.path.join(model_dir, f"{participant}_raw_model.h5")
        model.save(model_save_path)
        print(f"Model saved at {model_save_path}")

        # 결과 저장
        result_save_path = os.path.join(result_dir, f"{participant}_classification_report.txt")
        with open(result_save_path, "w") as f:
            f.write(report)
        print(f"Results saved at {result_save_path}")

# 메인 실행

def main():
    base_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG/4s_seg_conv_ch_BPF"
    base2_dir = "/home/bcml1/sigenv/4s_seg_intra_save"
    data_dir = base_dir
    result_dir = os.path.join(base2_dir, "results")
    model_dir = os.path.join(base2_dir, "models")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_intra_subject(data_dir, result_dir, model_dir)

if __name__ == "__main__":
    main()
