import numpy as np

data = np.load('/home/bcml1/2025_EMOTION/DEAP_numeric_labels/s01_emotion_labels.npy', allow_pickle=True)
print("데이터:", data)
print("데이터 타입:", data.dtype)
print("배열 형태:", data.shape)
