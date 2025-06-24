import os
import numpy as np
import scipy.io as sio
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler

def load_eeg_mat_file(file_path):
    """ .mat 파일에서 EEG 데이터를 로드 """
    mat_data = sio.loadmat(file_path)
    # 메타 데이터 제외하고 EEG 데이터만 선택 (예: 키가 '__'로 시작하지 않는 항목)
    eeg_data = {key: value for key, value in mat_data.items() if not key.startswith('__')}
    return eeg_data


def preprocess_eeg_data(eeg_data, apply_smoothing=False):
    """ EEG 데이터 전처리 (NaN 처리, 정규화, 스무딩 적용 가능) """
    for key in eeg_data:
        raw_values = eeg_data[key]
        if np.any(np.isnan(raw_values)):
            mean_values = np.nanmean(raw_values, axis=0)
            inds = np.where(np.isnan(raw_values))
            raw_values[inds] = np.take(mean_values, inds[1])

        # 정규화
        scaler = MinMaxScaler()
        eeg_data[key] = scaler.fit_transform(raw_values)

        # 스무딩 (옵션)
        if apply_smoothing:
            eeg_data[key] = medfilt(eeg_data[key], kernel_size=3)

    return eeg_data

def process_eeg_data(input_folder, output_folder, apply_smoothing=False):
    """ EEG 데이터 전처리 후 저장 """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mat'):
            file_path = os.path.join(input_folder, file_name)
            eeg_data = load_eeg_mat_file(file_path)
            processed_data = preprocess_eeg_data(eeg_data, apply_smoothing=apply_smoothing)

            output_file = os.path.join(output_folder, file_name.replace('.mat', '.npy'))
            np.save(output_file, processed_data)
            print(f"Processed EEG: {file_name} -> {output_file} (Smoothing: {apply_smoothing})")

# 실행 예시
input_eeg_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/3"
output_eeg_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/3_npy"
process_eeg_data(input_eeg_dir, output_eeg_dir, apply_smoothing=True)
