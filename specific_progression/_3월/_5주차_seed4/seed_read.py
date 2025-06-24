import numpy as np

data = np.load('/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE/2/1_20161125_sample_02_label_1.npy', allow_pickle=True)
print("데이터:", data)
print("데이터 타입:", data.dtype)
print("배열 형태:", data.shape)
