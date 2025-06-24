import os

eeg_npy_path = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG/s01_eeg.npy"
labels_npy_path = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_label/s01_labels.npy"

# 파일이 존재하는지 확인
print("EEG File Exists:", os.path.exists(eeg_npy_path))
print("Labels File Exists:", os.path.exists(labels_npy_path))
