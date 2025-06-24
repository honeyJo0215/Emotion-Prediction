import numpy as np

subject_path = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/1_npy/1_20160518.npy'
subject_data = np.load(subject_path, allow_pickle=True).item()
print("Loaded keys:", subject_data.keys())
