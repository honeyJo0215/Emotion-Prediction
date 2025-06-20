import os
import numpy as np

def combine_band_timestemp_channel(eeg_data, de_features):
    """
    Combine EEG and DE data into shape (band, 513, channel).

    Parameters:
    - eeg_data (np.ndarray): EEG data of shape (4(band), 512(timestemp), 8(channel)).
    - de_features (np.ndarray): DE features of shape (4(band), 8(channel)).

    Returns:
    - combined_data (np.ndarray): Combined data of shape (4(band), 513, 8(channel)).
    """
    # Expand DE features to add a time dimension (1)
    de_expanded = de_features[:, np.newaxis, :]  # (4, 1, 8)

    # Concatenate along the time axis (axis=1)
    combined_data = np.concatenate((eeg_data, de_expanded), axis=1)  # (4, 513, 8)

    return combined_data

def process_and_save_combined_data(test_dir, train_dir, de_dir, output_test_dir, output_train_dir):
    """
    Process and combine EEG and DE data, then save the combined files in appropriate directories.

    Parameters:
    - test_dir (str): Path to the test EEG data directory (s01).
    - train_dir (str): Path to the train EEG data directory.
    - de_dir (str): Path to the directory containing DE features.
    - output_test_dir (str): Directory to save combined test data.
    - output_train_dir (str): Directory to save combined train data.

    Returns:
    - None
    """
    
    def convert_eeg_to_de_name(eeg_filename):
        """
        Convert EEG filename to corresponding DE filename.

        Parameters:
        - eeg_filename (str): Filename of the EEG file.

        Returns:
        - de_filename (str): Corresponding DE filename.
        """
        # Replace `_segment_` with `_FB_segment_` and append `_de_features` before `.npy`
        updated_filename = eeg_filename.replace("_segment_", "_FB_segment_").replace("_FB.npy", "_de_features.npy")
        return updated_filename


    def save_combined(eeg_path, de_path, output_dir):
        # Load the data
        eeg_data = np.load(eeg_path)  # EEG data: (4, 512, 8)
        de_features = np.load(de_path)  # DE features: (4, 8)

        # Combine the data
        combined_data = combine_band_timestemp_channel(eeg_data, de_features)

        # Generate output path and save
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(eeg_path))
        np.save(output_path, combined_data)
        print(f"Combined data saved to: {output_path}")

    # Process EEG files in the test directory
    for filename in os.listdir(test_dir):
        if filename.endswith('.npy'):
            eeg_path = os.path.join(test_dir, filename)
            de_filename = convert_eeg_to_de_name(filename)
            de_path = os.path.join(de_dir, de_filename)

            if os.path.exists(de_path):
                save_combined(eeg_path, de_path, output_test_dir)
            else:
                print(f"DE file not found for {filename}")

    # Process EEG files in the train directory
    for filename in os.listdir(train_dir):
        if filename.endswith('.npy'):
            eeg_path = os.path.join(train_dir, filename)
            de_filename = convert_eeg_to_de_name(filename)
            de_path = os.path.join(de_dir, de_filename)

            if os.path.exists(de_path):
                save_combined(eeg_path, de_path, output_train_dir)
            else:
                print(f"DE file not found for {filename}")

# Example usage
process_and_save_combined_data(
    test_dir="/home/bcml1/2025_EMOTION/DEAP_EEG/overlap4s_0.125_seg_conv_ch_BPF/segment_sample_2D/test",
    train_dir="/home/bcml1/2025_EMOTION/DEAP_EEG/overlap4s_0.125_seg_conv_ch_BPF/segment_sample_2D/train",
    de_dir="/home/bcml1/sigenv/4s_DE_feature/de_features_overlap_0.125s",
    output_test_dir="/home/bcml1/2025_EMOTION/DEAP_EEG/overlap4s_seg_conv_ch_BPF+DE/test_overlap0.125",
    output_train_dir="/home/bcml1/2025_EMOTION/DEAP_EEG/overlap4s_seg_conv_ch_BPF+DE/train_overlap0.125"
)
