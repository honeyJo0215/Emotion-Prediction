import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score, confusion_matrix
from scipy.signal import butter, filtfilt

# TensorFlow 버전 확인
print(f"TensorFlow version: {tf.__version__}")

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            try:
                # 최신 API 사용
                tf.config.set_memory_growth(gpu, True)
                print(f"Set memory growth for GPU: {gpu}")
            except AttributeError:
                # 이전 API 사용
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Set memory growth for GPU using experimental API: {gpu}")
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")
else:
    print("No GPU devices found. Running on CPU.")


# Step 1: 데이터 로드 함수
def load_eeg_data(directory):
    """
    Load raw EEG data and corresponding labels from the specified directory.

    Parameters:
        directory (str): Path to the directory containing .npy files.

    Returns:
        data_list (list): List of raw EEG data arrays.
        labels (list): List of labels corresponding to each sample.
        participants (list): List of participant IDs corresponding to each sample.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError("No .npy files found in the directory.")

    data_list, labels, participants = [], [], []
    for file in files:
        # Extract participant ID and label from filename
        basename = os.path.basename(file)
        parts = basename.split('_')
        if len(parts) < 3:
            raise ValueError(f"Unexpected file naming format: {basename}")
        
        participant_id = parts[0]
        label_str = parts[1]
        # 'FB'는 세션 식별자가 아니므로 무시

        if 'positive' in label_str:
            label = 1
        elif 'negative' in label_str:
            label = 0
        else:
            raise ValueError(f"File name does not indicate label: {file}")

        raw_data = np.load(file)

        # Check if raw_data has the expected shape (samples, channels, timesteps)
        if len(raw_data.shape) != 3:
            raise ValueError(f"Unexpected data shape {raw_data.shape} in file: {file}")

        # Append raw data and metadata
        data_list.append(raw_data)
        labels.extend([label] * raw_data.shape[0])
        participants.extend([participant_id] * raw_data.shape[0])

    return np.concatenate(data_list), np.array(labels), np.array(participants)

# Step 2: 밴드패스 필터링 함수
def bandpass_filter(data, lowcut=4, highcut=45, fs=128, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

# Step 3: 데이터 정규화 함수
def normalize_eeg_data(data):
    return (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)

# Step 4: CNN 모델 정의
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        
        Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        
        Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 5: 데이터 전처리 함수
def preprocess_eeg_data(directory):
    X, y, participants = load_eeg_data(directory)
    X = bandpass_filter(X)
    X = normalize_eeg_data(X)
    return X, y, participants

# Step 6: Inter-Subject Cross-Validation(inter-session cross-validation)을 사용한 Train-Test 분할 및 모델 학습
def train_model_train_test_split(X, y, participants, model_path, train_ratio=0.8):
    """
    Train the CNN model using a per-participant train-test split.
    After training, evaluate the model on the test data.

    Parameters:
        X (np.ndarray): Preprocessed feature data.
        y (np.ndarray): Corresponding labels.
        participants (np.ndarray): Participant IDs.
        model_path (str): Path to save the trained model.
        train_ratio (float): Ratio of training data per participant.
    """
    unique_participants = np.unique(participants)
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    for participant in unique_participants:
        participant_mask = (participants == participant)
        X_participant = X[participant_mask]
        y_participant = y[participant_mask]

        # Split per participant
        if len(X_participant) < 2:
            raise ValueError(f"Not enough samples for participant {participant} to perform train-test split.")
        
        # Stratified split to maintain class distribution
        X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(
            X_participant, y_participant, train_size=train_ratio, random_state=42, shuffle=True, stratify=y_participant
        )

        X_train_list.append(X_train_part)
        X_test_list.append(X_test_part)
        y_train_list.extend(y_train_part)
        y_test_list.extend(y_test_part)

    # Concatenate all train and test data
    X_train = np.concatenate(X_train_list)
    X_test = np.concatenate(X_test_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    print(f"Total training samples: {X_train.shape[0]}")
    print(f"Total testing samples: {X_test.shape[0]}")

    # One-hot encode labels
    num_classes = len(np.unique(y))
    y_train_categorical = to_categorical(y_train, num_classes)
    y_test_categorical = to_categorical(y_test, num_classes)

    input_shape = X_train.shape[1:]

    # Create and train the model
    model = create_cnn_model(input_shape, num_classes)
    
    # 조기 종료 콜백 추가 (선택 사항)
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, 
        y_train_categorical, 
        epochs=50, 
        batch_size=32, 
        verbose=1,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    # Evaluate the model on test data
    predictions = model.predict(X_test)
    test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)[1]
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test_categorical, predictions, multi_class='ovr')
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Calculate F1 score
    y_pred_labels = np.argmax(predictions, axis=1)
    f1 = f1_score(y_test, y_pred_labels, average='weighted')
    print(f"F1 Score: {f1:.4f}")

    # Calculate FPR, TPR
    cm = confusion_matrix(y_test, y_pred_labels)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    
    # Plot ROC curve
    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_categorical[:, i], predictions[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    # Save results
    report = classification_report(y_test, y_pred_labels, digits=4)
    print("Detailed Classification Report:")
    print(report)

    with open("bio_cnn_eeg_results_model2_re.txt", "w") as result_file:
        result_file.write("Train-Test Split Results\n")
        result_file.write("\nFinal Accuracy:\n")
        result_file.write(f"{test_accuracy:.4f}\n\n")
        result_file.write("Detailed Classification Report:\n")
        result_file.write(report)
        result_file.write(f"\nFalse Positive Rate: {fpr:.4f}\n")
        result_file.write(f"True Positive Rate: {tpr:.4f}\n")
        result_file.write("\nPrediction Details (file-level):\n")
        for participant in unique_participants:
            participant_mask = (participants == participant)
            X_participant = X_test[participant_mask]
            y_participant = y_test[participant_mask]
            y_pred_participant = y_pred_labels[participant_mask]

            for pred, actual in zip(y_pred_participant, y_participant):
                correctness = "Correct" if pred == actual else "Incorrect"
                result_file.write(f"Participant {participant}: Predicted={pred}, Actual={actual} ({correctness})\n")

    # Save the trained model
    model.save(model_path)
    print(f"Model saved at {model_path}")

# Main execution block
def main():
    # 데이터 경로 설정
    data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG/ch_BPF"  # 실제 데이터 디렉토리로 업데이트
    model_path = "cnn_model_inter_session.h5"  # 원하는 모델 저장 경로로 업데이트

    print("Preprocessing EEG data...")
    X, y, participants = preprocess_eeg_data(data_dir)

    print("Training the model...")
    train_model_train_test_split(X, y, participants, model_path, train_ratio=0.8)

if __name__ == "__main__":
    main()