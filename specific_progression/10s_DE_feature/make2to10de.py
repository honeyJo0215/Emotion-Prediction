# 2초 단위로 잘라서 만든 DE feature는 3차원이잖아? 이거를 1차원으로 변환해서 1차원으로 변환한 다섯개의 DE feature(2초 x 5개)를 1차원 배열으로 이어붙여서 10초를 가진 feature 하나로 만드는 작업을 반복하고 싶어. 이렇게해서 한 길이 당 10초인 1차원 배열을 만들게 되면, 이 논문에서 얘기하는 모델을 적용할 수 있지 않을까? ->에서 시작되는 feature 만들기.

import numpy as np
import os

def load_and_process_de_features(input_dir, output_dir, window_size=5, slide_step=1):
    """
    Load DE features, convert to 1D, and create 10-second segments using sliding window.

    Parameters:
        input_dir: str
            Directory containing the DE feature files.
        output_dir: str
            Directory to save the processed 10-second segments.
        window_size: int
            Number of 2-second DE features to concatenate for 10 seconds.
        slide_step: int
            Step size for sliding window (number of 2-second features to slide).
    """
    os.makedirs(output_dir, exist_ok=True)

    # List all DE feature files
    input_files = [f for f in os.listdir(input_dir) if f.endswith('_de_features.npy')]

    for file in input_files:
        input_path = os.path.join(input_dir, file)
        print(f"Processing file: {input_path}")

        # Load DE feature data
        de_features = np.load(input_path)  # Shape: (samples, bands, channels)
        n_samples, n_bands, n_channels = de_features.shape

        # Flatten each DE feature (bands x channels → 1D)
        flattened_features = de_features.reshape(n_samples, -1)  # Shape: (samples, bands*channels)

        # Create 10-second segments using sliding window
        concatenated_features = []
        for start_idx in range(0, n_samples - window_size + 1, slide_step):
            end_idx = start_idx + window_size
            segment = flattened_features[start_idx:end_idx].flatten()  # Concatenate into 1D
            concatenated_features.append(segment)

        concatenated_features = np.array(concatenated_features)  # Shape: (n_segments, window_size*bands*channels)

        # Save the processed features
        output_file = os.path.join(output_dir, file.replace('_de_features.npy', '_10s_features.npy'))
        np.save(output_file, concatenated_features)
        print(f"Saved 10-second features to {output_file}")

# Input and output directories
input_dir = "/home/bcml1/sigenv/2s_DE_feature/de_features"
output_dir = "./10s_processed_features"

# Process DE features and create 10-second segments using sliding window
load_and_process_de_features(input_dir, output_dir, window_size=5, slide_step=1)