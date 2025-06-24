#표준화 (Standardization)가 Attention에 잘맞기에, DE feature을 추출할 때 이 표준화를 사용한다.

import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import logging
from multiprocessing import Pool

# 로그 설정
logging.basicConfig(filename='process_eeg.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

def calculate_de(segment):
    variance = np.var(segment)
    if variance == 0:
        variance = 1e-6  # 로그(0) 방지
    return 0.5 * np.log(2 * np.pi * np.e * variance)

def standardize_data_vectorized(eeg_data):
    bands, timestamps, channels = eeg_data.shape
    reshaped_data = eeg_data.reshape(bands * channels, timestamps)
    scaler = StandardScaler()
    standardized_reshaped = scaler.fit_transform(reshaped_data)
    standardized_data = standardized_reshaped.reshape(bands, channels, timestamps).transpose(0, 2, 1)
    return standardized_data

def extract_de_features(eeg_data, standardization=True):
    if eeg_data.ndim != 3:
        raise ValueError(f"EEG 데이터의 차원이 3이 아닙니다: {eeg_data.shape}")

    bands, timestamps, channels = eeg_data.shape

    if standardization:
        eeg_data = standardize_data_vectorized(eeg_data)

    features = np.zeros((bands, channels))
    for band in range(bands):
        for channel in range(channels):
            segment = eeg_data[band, :, channel]
            de = calculate_de(segment)
            features[band, channel] = de

    return features

def process_single_file(file, input_dir, output_dir):
    input_path = os.path.join(input_dir, file)
    logging.info(f"Processing file: {input_path}")

    try:
        eeg_data = np.load(input_path)
    except Exception as e:
        logging.error(f"Error loading file {file}: {e}")
        return

    logging.info(f"File {file} shape: {eeg_data.shape}")

    if eeg_data.ndim != 3:
        logging.warning(f"File {file} has invalid shape {eeg_data.shape}. Skipping.")
        return

    try:
        de_features = extract_de_features(eeg_data, standardization=True)
    except Exception as e:
        logging.error(f"Error extracting DE from file {file}: {e}")
        return

    base_name = os.path.basename(file).replace(".npy", "")
    parts = base_name.split("_")
    
    if len(parts) < 6:
        logging.warning(f"Filename {file} does not match expected format. Skipping.")
        return

    subject = parts[0]
    sample = parts[2]
    segment = parts[4]
    condition = parts[5]

    output_file_name = f"{subject}_sample_{sample}_segment_{segment}_{condition}_DE_standardized.npy"
    output_file = os.path.join(output_dir, output_file_name)
    
    try:
        np.save(output_file, de_features)
        logging.info(f"Saved DE features to {output_file}")
    except Exception as e:
        logging.error(f"Error saving file {output_file}: {e}")

def process_eeg_files_parallel(input_dir, output_dir, num_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    with Pool(processes=num_workers) as pool:
        pool.starmap(process_single_file, [(file, input_dir, output_dir) for file in input_files])

# 실행 예제
if __name__ == "__main__":
    input_dir = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_sam_seg_labeled"
    output_dir = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_de_features"
    process_eeg_files_parallel(input_dir, output_dir, num_workers=8)  # 필요에 따라 워커 수 조정
