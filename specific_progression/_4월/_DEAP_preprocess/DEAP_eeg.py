import os
import numpy as np

# Define input and output directories
input_folder = '/home/bcml1/2025_EMOTION/DEAP_eeg_npy_files'
output_folder = '/home/bcml1/2025_EMOTION/DEAP_eeg_new'

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over each subject (from s01_signals.npy to s32_signals.npy)
for subj in range(1, 33):
    # Construct the file name for the subject, e.g., "s01_signals.npy"
    subject_filename = f's{subj:02d}_signals.npy'
    subject_filepath = os.path.join(input_folder, subject_filename)
    
    # Load the subject's data (expected shape: (40, 32, T) where T is 8064)
    data = np.load(subject_filepath)
    
    # Loop through each sample (first dimension: 40 samples)
    for sample_idx in range(data.shape[0]):
        # Construct output file name in the format:
        # folder1_subject{subject number}_sample_{sample number}.npy
        # Example: "folder1_subject1_sample_01.npy"
        out_filename = f'folder1_subject{subj}_sample_{sample_idx+1:02d}.npy'
        out_filepath = os.path.join(output_folder, out_filename)
        
        # Extract the sample data (shape: (32, T))
        sample_data = data[sample_idx]
        
        # Save the sample data as a separate npy file
        np.save(out_filepath, sample_data)
        
        print(f"Saved: {out_filepath}")
