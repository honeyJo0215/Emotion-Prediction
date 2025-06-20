import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# TensorFlow 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Failed to set memory growth: {e}")
    else:
        print("No GPU devices found. Running on CPU.")

configure_gpu()

def load_data(directory):
    files_positive = []
    files_negative = []

    for file in os.listdir(directory):
        if '_positive_' in file:
            files_positive.append(os.path.join(directory, file))
        elif '_negative_' in file:
            files_negative.append(os.path.join(directory, file))

    if not files_positive or not files_negative:
        raise ValueError(f"No files found. Positive: {files_positive}, Negative: {files_negative}")

    positive_data = [np.load(file) for file in files_positive]
    negative_data = [np.load(file) for file in files_negative]

    X = np.concatenate(positive_data + negative_data, axis=0)
    y = np.array([1] * len(np.concatenate(positive_data)) + [0] * len(np.concatenate(negative_data)))

    participants = []
    for file in files_positive + files_negative:
        participant_id = os.path.basename(file).split('_')[0]
        participants.extend([participant_id] * np.load(file).shape[0])

    participants = np.array(participants)
    return X, y, participants

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2, strides=1),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2, strides=1),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predict_test_samples():
    directory = '/home/bcml1/2025_EMOTION/DEAP_EEG/ch_BPF'  # 데이터를 저장한 디렉토리 경로
    X, y, participants = load_data(directory)

    print(f"X shape: {X.shape}, y shape: {y.shape}, participants shape: {participants.shape}")

    if participants.shape[0] != X.shape[0] or participants.shape[0] != y.shape[0]:
        raise ValueError("Mismatch in dimensions: 'participants', 'X', and 'y' must have the same number of samples.")

    logo = LeaveOneGroupOut()
    accuracy_scores = []

    for train_index, test_index in logo.split(X, y, groups=participants):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = create_cnn_model(X.shape[1:])
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracy_scores.append(test_accuracy)

        print(f"Test session accuracy: {test_accuracy:.4f}")

    print(f"Mean accuracy across sessions: {np.mean(accuracy_scores):.4f}")

if __name__ == "__main__":
    predict_test_samples()
