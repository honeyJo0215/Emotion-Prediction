import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score

# Step 0: Preprocess EEG-only data (filter out labels from raw data)
def preprocess_eeg_data(directory):
    """
    Load and preprocess EEG data only, excluding any label patterns.
    Includes normalization and participant-specific segmentation.

    Parameters:
        directory (str): Path to the directory containing .npy files.

    Returns:
        data (np.ndarray): Normalized EEG data array.
        labels (np.ndarray): Corresponding labels for each sample.
        participants (np.ndarray): Participant IDs.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError("No .npy files found in the directory.")

    data, labels, participants = [], [], []
    for file in files:
        # Extract participant ID and labels from filename
        participant_id = os.path.basename(file).split('_')[0]
        if 'positive' in file:
            label = 1
        elif 'negative' in file:
            label = 0
        else:
            raise ValueError(f"File name does not indicate label: {file}")

        raw_data = np.load(file)

        # Check if raw_data has the expected shape (samples, channels, timesteps)
        if len(raw_data.shape) != 3:
            raise ValueError(f"Unexpected data shape {raw_data.shape} in file: {file}")

        # Extract EEG data and normalize per sample and channel
        normalized_data = (raw_data - np.mean(raw_data, axis=-1, keepdims=True)) / np.std(raw_data, axis=-1, keepdims=True)

        # Append processed data and metadata
        data.append(normalized_data)
        labels.extend([label] * normalized_data.shape[0])
        participants.extend([participant_id] * normalized_data.shape[0])

    # Merge data
    return np.concatenate(data), np.array(labels), np.array(participants)

# Step 1: Define CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 2: Train and save the model
def train_model(X, y, participants, model_path):
    """
    Train the CNN model using preprocessed EEG-only data and save it.

    Parameters:
        X (np.ndarray): Preprocessed feature data.
        y (np.ndarray): Corresponding labels.
        participants (np.ndarray): Participant IDs for Leave-One-Group-Out validation.
        model_path (str): Path to save the trained model.
    """
    logo = LeaveOneGroupOut()
    input_shape = X.shape[1:]
    num_classes = len(np.unique(y))
    model = create_cnn_model(input_shape, num_classes)

    accuracy_scores = []
    for train_index, test_index in logo.split(X, y, groups=participants):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Convert labels to categorical
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Evaluate the model
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
        accuracy_scores.append(test_accuracy)

    print(f"Average accuracy across folds: {np.mean(accuracy_scores):.4f}")
    model.save(model_path)
    print(f"Model saved at {model_path}")

# Step 3: Load and evaluate saved model
def evaluate_model(X, y, participants, model_path):
    model = load_model(model_path)
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []

    for train_index, test_index in logo.split(X, y, groups=participants):
        X_test = X[test_index]
        y_test = y[test_index]

        # Convert labels to categorical
        y_test_categorical = to_categorical(y_test, num_classes=len(np.unique(y)))

        # Predict
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Collect results
        all_true.extend(y_test)
        all_pred.extend(y_pred)

    # Final evaluation
    print(classification_report(all_true, all_pred, digits=4))

# Main execution block
def main():
    data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG/ch_BPF"  # Update with actual data directory
    model_path = "cnn_model_eeg_only.h5"  # Update with desired model save path

    print("Preprocessing EEG data...")
    X, y, participants = preprocess_eeg_data(data_dir)

    print("Training the model...")
    train_model(X, y, participants, model_path)

    print("Evaluating the model...")
    evaluate_model(X, y, participants, model_path)

if __name__ == "__main__":
    main()
