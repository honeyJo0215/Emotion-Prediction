import os
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

# Data path and result storage path
data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPS3'
save_dir = '/home/bcml1/sigenv/_4주차_data_check/DEAP_PPS_Visualize'
os.makedirs(save_dir, exist_ok=True)

# Subject list: s01 ~ s22
subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
num_trials = 40          # trial_00 ~ trial_39
num_channels = 8         # channel 33~40

# Processing by each subject
for subject in subjects:
    cols = 7
    rows = math.ceil(num_trials / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), sharex=True)
    axes = axes.flatten()
    
    for trial in range(num_trials):
        ax = axes[trial]
        # File name pattern: sXX_trial_YY_label_*.npy (label is encoded in the file name)
        pattern = os.path.join(data_dir, f"{subject}_trial_{trial:02d}_label_*.npy")
        matching_files = glob.glob(pattern)
        if not matching_files:
            print(f"File not found for {subject} trial {trial:02d}")
            continue
        
        # Use the first file if there are multiple files in that trial
        file_path = matching_files[0]
        data = np.load(file_path)
        
        # Process both shapes: (8, 128) or (60, 8, 128)
        if data.ndim == 2 and data.shape == (num_channels, 128):
            # If you are already in (8, 128) form
            processed_data = data
            time_vector = np.linspace(0, 60, 128)
        elif data.ndim == 3 and data.shape == (60, num_channels, 128):
            # (60, 8, 128) If in form: 60 segments attached together
            # Reshape after first changing the axis order to (60, 8, 128) -> (8, 60, 128)
            processed_data = data.transpose(1, 0, 2).reshape(num_channels, -1)
            time_vector = np.linspace(0, 60, processed_data.shape[1])
        else:
            print(f"Unexpected shape in {file_path}: {data.shape}")
            continue
        
        # Plot each channel (33 to 40) signal
        for ch in range(num_channels):
            ax.plot(time_vector, processed_data[ch, :], label=f"Ch {ch+33}", linewidth=1)
        ax.set_title(f"Trial {trial:02d}", fontsize=10)
        if trial % cols == 0:
            ax.set_ylabel("Signal", fontsize=8)
        if trial >= num_trials - cols:
            ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize='x-small', loc='upper right')
    
    # Delete the remaining subplot
    for i in range(num_trials, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(f"Subject {subject} Trials Visualization", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"{subject}_trials_visualization.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization for {subject} at {save_path}")
