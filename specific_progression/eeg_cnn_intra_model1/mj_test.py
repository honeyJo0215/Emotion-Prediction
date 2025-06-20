import numpy as np
from sklearn.model_selection import LeaveOneOut
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# TensorFlow 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")
else:
    print("No GPU devices found. Running on CPU.")

def load_data():
    # 데이터 파일 로드
    data_dir = "/home/bcml1/sigenv/eeg_band_split"
    
    # 파일 로드 및 병합
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    X_list, y_list, participants_list = [], [], []

    for file in files:
        try:
            data = np.load(os.path.join(data_dir, file))
            
            # 파일별로 X, y, participants 추가
            X_list.append(data)

            # 파일명에서 참여자와 라벨 추출
            participant = file.split('_')[0]  # 예: s01
            label = file.split('_')[1]       # 예: negative

            participants_list.extend([participant] * data.shape[0])  # 참여자 반복
            y_list.extend([label] * data.shape[0])  # 라벨 반복
        except Exception as e:
            print(f"Error loading file {file}: {e}")

    # 배열로 변환. 병합
    X = np.concatenate(X_list, axis=0)  # 데이터를 하나의 배열로 병합
    y = np.array(y_list)
    participants = np.array(participants_list)
    
    # 크기 확인
    if len(X) != len(y) or len(X) != len(participants):
        raise ValueError(f"Mismatch in data lengths: X={len(X)}, y={len(y)}, participants={len(participants)}")

    # Conv1D 입력 형태로 변환
    if len(X.shape) == 2:  # (samples, time_steps)
        X = np.expand_dims(X, axis=-1)  # (samples, time_steps, 1)

    return X, y, participants

# CNN 모델 정의
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),  # Conv1D에 맞는 입력 형태
        Conv1D(32, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),

        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 테스트 데이터 예측 함수
def predict_test_samples():
    X, y, participants = load_data()

    # 데이터 크기 확인
    print(f"X shape: {X.shape}, y shape: {y.shape}, participants shape: {participants.shape}")

    # participants 배열의 크기가 X와 y의 첫 번째 축과 일치하지 않는 경우 처리
    if participants.shape[0] != X.shape[0] or participants.shape[0] != y.shape[0]:
        raise ValueError("Mismatch in dimensions: 'participants', 'X', and 'y' must have the same number of samples.")

    # 문자열 라벨을 숫자형 라벨로 매핑
    unique_labels = np.unique(y)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    y_int = np.array([label_to_int[label] for label in y])

    num_classes = len(unique_labels)
    input_shape = X.shape[1:]  # 데이터가 올바르게 reshape되었다고 가정
    y_categorical = to_categorical(y_int, num_classes=num_classes)

    loo = LeaveOneOut()
    accuracy_scores = []

    for train_index, test_index in loo.split(np.unique(participants)):
        train_participants = np.unique(participants)[train_index]
        test_participant = np.unique(participants)[test_index][0]

        # 참여자별 데이터 분리
        train_mask = np.isin(participants, train_participants)
        test_mask = participants == test_participant

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_categorical[train_mask], y_categorical[test_mask]

        # 모델 생성 및 학습
        model = create_cnn_model(input_shape, num_classes)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

        # 학습 과정 확인
        print("Training history:")
        print("Accuracy:", history.history['accuracy'])
        print("Loss:", history.history['loss'])

        # 모델 평가
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracy_scores.append(test_accuracy)

        print(f"Test participant: {test_participant}, Test accuracy: {test_accuracy:.4f}")

        # 테스트 데이터 예측
        for i, test_sample in enumerate(X_test):
            test_sample_expanded = np.expand_dims(test_sample, axis=0)  # 배치 차원 추가
            prediction = model.predict(test_sample_expanded)
            predicted_class = np.argmax(prediction, axis=1)[0]  # 예측된 클래스
            true_class = np.argmax(y_test[i])  # 실제 클래스
            print(f"Sample {i}: Prediction (0=negative, 1=positive): {predicted_class}, True: {true_class}")

    print(f"Mean accuracy across participants: {np.mean(accuracy_scores):.4f}")

if __name__ == "__main__":
    predict_test_samples()
