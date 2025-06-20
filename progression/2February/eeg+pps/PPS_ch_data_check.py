import os
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

# The path where the data is located and the path where the results are stored
data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPS3'
save_dir = '/home/bcml1/sigenv/_4주차_data_check/DEAP_PPS_ch_Visualize'
os.makedirs(save_dir, exist_ok=True)

# Subject List (s01 ~ s22)
subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
num_trials = 40          # trial_00 ~ trial_39
num_channels = 8         # channel 33 ~ 40

# Processing for each subject
for subject in subjects:
    # Visualization for each channel (8 images in total)
    for ch in range(num_channels):
        # subplot grid: Calculate rows in 7 columns to match the number of trials
        cols = 7
        rows = math.ceil(num_trials / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), sharex=True)
        axes = axes.flatten()

        # Plot the corresponding channel signal for each trial
        for trial in range(num_trials):
            ax = axes[trial]
            # File name pattern: sXX_trial_YY_label_*.npy (label is encoded)
            pattern = os.path.join(data_dir, f"{subject}_trial_{trial:02d}_label_*.npy")
            matching_files = glob.glob(pattern)
            if not matching_files:
                print(f"File not found for {subject} trial {trial:02d}")
                continue

            # Use the first file if you have multiple files
            file_path = matching_files[0]
            data = np.load(file_path)

            # Processing by data shape
            if data.ndim == 2 and data.shape == (num_channels, 128):
                # If (8, 128), select the corresponding channel data
                channel_data = data[ch, :]
                time_vector = np.linspace(0, 60, 128)
            elif data.ndim == 3 and data.shape == (60, num_channels, 128):
                # If (60, 8, 128) connect segments to form a single signal
                processed_data = data.transpose(1, 0, 2).reshape(num_channels, -1)
                channel_data = processed_data[ch, :]
                time_vector = np.linspace(0, 60, processed_data.shape[1])
            else:
                print(f"Unexpected shape in {file_path}: {data.shape}")
                continue

            ax.plot(time_vector, channel_data, linewidth=1)
            ax.set_title(f"Trial {trial:02d}", fontsize=10)
            if trial % cols == 0:
                ax.set_ylabel("Signal", fontsize=8)
            if trial >= num_trials - cols:
                ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=8)
        
        # Delete unused subplot
        for i in range(num_trials, len(axes)):
            fig.delaxes(axes[i])
        
        fig.suptitle(f"Subject {subject} - Channel {ch+33} Visualization", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"{subject}_channel_{ch+33}_trials_visualization.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved visualization for {subject} channel {ch+33} at {save_path}")
