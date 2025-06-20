import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import LeaveOneGroupOut

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #이것을 위로 두지 않으면 GPU 대신 CPU로 설정하는 게 동작하지 않음
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

def load_data(directory):
    # Positive(긍정)과 Negative(부정) 데이터를 저장할 리스트 초기화
    files_positive = []
    files_negative = []

    # 디렉토리에 있는 파일들을 탐색
    for file in os.listdir(directory):
        # 파일 이름에 '_positive_'가 포함되어 있으면 positive 파일 리스트에 추가
        if '_positive_' in file:
            files_positive.append(os.path.join(directory, file))
        # 파일 이름에 '_negative_'가 포함되어 있으면 negative 파일 리스트에 추가
        elif '_negative_' in file:
            files_negative.append(os.path.join(directory, file))

    # positive 또는 negative 파일이 하나도 없을 경우 예외를 발생시킴
    if not files_positive or not files_negative:
        raise ValueError(f"No files found. Positive: {files_positive}, Negative: {files_negative}")

    # positive와 negative 파일의 데이터를 읽어서 각각 리스트에 저장
    positive_data = [np.load(file) for file in files_positive]  # positive 파일 데이터 로드
    negative_data = [np.load(file) for file in files_negative]  # negative 파일 데이터 로드

    # positive와 negative 데이터를 합쳐서 X 데이터로 생성
    # axis=0은 데이터를 샘플 축(행)으로 결합함
    X = np.concatenate(positive_data + negative_data, axis=0)

    # y 레이블 생성
    # positive 데이터의 개수만큼 1로 된 레이블 생성 후
    # negative 데이터의 개수만큼 0으로 된 레이블을 추가
    y = np.array([1] * len(np.concatenate(positive_data)) + [0] * len(np.concatenate(negative_data)))

    # 참여자 정보(participants) 생성
    # 각 파일 이름에서 참여자 ID를 추출하고, 파일에 있는 데이터 개수만큼 반복
    participants = []
    for file in files_positive + files_negative:
        # 파일 이름에서 참여자 ID 추출 (파일 이름은 예: "001_positive_data.npy")
        participant_id = os.path.basename(file).split('_')[0]
        # np.load(file).shape[0]: 해당 파일에 있는 샘플 개수
        participants.extend([participant_id] * np.load(file).shape[0])

    # participants 리스트를 NumPy 배열로 변환
    participants = np.array(participants)

    # 최종적으로 데이터(X), 레이블(y), 참여자 정보(participants)를 반환
    return X, y, participants

def create_cnn_model(input_shape, num_classes):
    """
    CNN 모델을 생성하고 컴파일하는 함수

    Parameters:
        input_shape (tuple): 입력 데이터의 형태 (예: (timesteps, features))
        num_classes (int): 출력 클래스의 수 (예: 감정 레이블의 개수)

    Returns:
        model (Sequential): 생성된 CNN 모델
    """
    # Sequential 모델 생성
    model = Sequential([
        # 1D Convolutional Layer: 32개의 필터, kernel_size=3, 활성화 함수 ReLU
        # input_shape 지정 (첫 번째 레이어에서만 필요)
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),

        # MaxPooling Layer: 풀링 크기(pool_size)=2, strides=1
        # 특징 추출 후 차원을 줄여 모델의 복잡도를 감소시킴
        MaxPooling1D(pool_size=2, strides=1),

        # 1D Convolutional Layer: 64개의 필터, kernel_size=3, 활성화 함수 ReLU
        Conv1D(64, kernel_size=3, activation='relu'),

        # MaxPooling Layer: 풀링 크기(pool_size)=2, strides=1
        MaxPooling1D(pool_size=2, strides=1),

        # Flatten Layer: 1D 데이터를 1차원으로 펼침
        # Fully Connected Layer에 입력으로 전달
        Flatten(),

        # Fully Connected Layer: 128개의 뉴런, 활성화 함수 ReLU
        Dense(128, activation='relu'),

        # Dropout Layer: 50%의 뉴런을 무작위로 비활성화 (overfitting 방지)
        Dropout(0.5),

        # Output Layer: 클래스 수에 따라 Softmax 활성화 함수 사용
        # 클래스별 확률 분포 출력
        Dense(num_classes, activation='softmax')
    ])

    # 모델 컴파일: 최적화 알고리즘, 손실 함수, 평가지표 설정
    # optimizer: Adam, 손실 함수: categorical_crossentropy, 평가지표: accuracy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model  # 생성된 모델 반환

# 테스트 데이터 예측 함수
def predict_test_samples():
    directory = '/home/bcml1/2025_EMOTION/DEAP_EEG/ch_BPF'  # 데이터를 저장한 디렉토리 경로
    #directory = '/home/bcml1/sigenv/eeg_cnn_model1'  # 데이터를 저장한 디렉토리 경로
    X, y, participants = load_data(directory)

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

    # Leave-One-Group-Out 방식으로 intra-session 검증
    logo = LeaveOneGroupOut()
    accuracy_scores = []

    for train_index, test_index in logo.split(X, y_categorical, groups=participants):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_categorical[train_index], y_categorical[test_index]

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

        print(f"Test session accuracy: {test_accuracy:.4f}")

    print(f"Mean accuracy across sessions: {np.mean(accuracy_scores):.4f}")

# 텍스트로 저장하는 코드 구현
if __name__ == "__main__":
    predict_test_samples()