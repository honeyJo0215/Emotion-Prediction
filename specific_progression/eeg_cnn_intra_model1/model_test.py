import os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report

def preprocess_test_data(directory):
    """
    Preprocess test data from the specified directory.
    Assumes files are in .npy format with samples and channels.

    Parameters:
        directory (str): Path to the directory containing .npy files.

    Returns:
        data (list): List of preprocessed data arrays.
        file_names (list): List of file names corresponding to the data.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError("No .npy files found in the directory.")

    data = []
    file_names = []

    for file in files:
        raw_data = np.load(file)

        # Normalize data per channel
        normalized_data = (raw_data - np.mean(raw_data, axis=-1, keepdims=True)) / np.std(raw_data, axis=-1, keepdims=True)
        data.extend([normalized_data[i] for i in range(normalized_data.shape[0])])
        file_names.extend([file] * normalized_data.shape[0])

    return data, file_names

def test_model_on_samples(model_path, test_data_dir):
    """
    Load the saved model and test it on preprocessed samples.

    Parameters:
        model_path (str): Path to the saved .h5 model file.
        test_data_dir (str): Directory containing .npy files for testing.
    """
    # Load the saved model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Preprocess the test data
    test_data, file_names = preprocess_test_data(test_data_dir)

    predictions = []
    for i, sample in enumerate(test_data):
        sample = sample[np.newaxis, :, :]  # Reshape for model input (1, channels, timesteps)
        prediction_prob = model.predict(sample)
        prediction = np.argmax(prediction_prob, axis=1)
        predictions.append(prediction[0])
        print(f"File: {file_names[i]} - Predicted Label: {prediction[0]} - Probabilities: {prediction_prob[0]}")

    # If ground truth labels are available, calculate metrics
    # Assuming labels can be extracted or provided externally
    # labels = ... (Load corresponding ground truth labels)
    # print(classification_report(labels, predictions, digits=4))

# Example usage
if __name__ == "__main__":
    model_file_path = "cnn_intra_model.h5"  # Path to the saved model
    test_data_directory = "/home/bcml1/2025_EMOTION/DEAP_EEG/ch_BPF"  # Directory containing test .npy files

    test_model_on_samples(model_file_path, test_data_directory)
