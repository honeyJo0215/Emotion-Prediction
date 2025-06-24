import os
import glob
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 전처리 및 저장을 위한 폴더 생성 함수
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 데이터 경로 설정 (필요에 따라 수정)
seed_iv_root = '/home/bcml1/2025_EMOTION/SEED_IV'  # SEED-IV 데이터 셋의 루트 폴더
eeg_raw_path = os.path.join(seed_iv_root, 'eeg_raw_data')
eye_raw_path = os.path.join(seed_iv_root, 'eye_raw_data')

# 전처리 후 저장할 폴더
processed_eeg_path = os.path.join(seed_iv_root, 'processed_eeg_raw_data')
processed_eye_path = os.path.join(seed_iv_root, 'processed_eye_raw_data')
ensure_dir(processed_eeg_path)
ensure_dir(processed_eye_path)

# 예시: EEG 데이터 전처리
def preprocess_eeg(mat_data):
    """
    예시 전처리 함수.
    mat_data: .mat 파일로부터 읽어들인 dict
    return: 전처리된 numpy 배열 (예: 각 trial의 평균 또는 간단한 normalization)
    """
    # .mat 파일은 보통 trial별 데이터를 저장하는 key (예: 'cz_eeg1', 'cz_eeg2', ...)를 가짐
    # 여기서는 첫 번째 key의 데이터를 예시로 사용합니다.
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if not keys:
        return None
    data = mat_data[keys[0]]
    # 간단한 normalization (평균 0, 표준편차 1)
    data = (data - np.mean(data)) / np.std(data)
    return data

# EEG .mat 파일 전처리 및 저장
eeg_files = glob.glob(os.path.join(eeg_raw_path, '*', '*.mat'))
for file in eeg_files:
    mat = sio.loadmat(file)
    proc_data = preprocess_eeg(mat)
    if proc_data is not None:
        # 저장 파일명: 원본 파일명과 동일하게 .npy 형식으로 저장
        base_name = os.path.basename(file).replace('.mat', '.npy')
        save_path = os.path.join(processed_eeg_path, base_name)
        np.save(save_path, proc_data)
        print(f"Processed EEG saved: {save_path}")

# 예시: Eye movement 데이터 전처리
def preprocess_eye(mat_data):
    """
    간단한 전처리: 각 행렬의 기본 통계치를 계산하는 예시 함수.
    mat_data: .mat 파일에서 읽은 데이터
    return: dictionary with mean, std for each trial (assuming list/array of trials)
    """
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    processed = {}
    for k in keys:
        trial = mat_data[k]
        processed[k] = {
            'mean': np.mean(trial),
            'std': np.std(trial),
            'min': np.min(trial),
            'max': np.max(trial)
        }
    return processed

# eye_raw_data 폴더 내 각 파일들에 대해 (각 eye modality 파일: blink, fixation, etc.)
eye_files = glob.glob(os.path.join(eye_raw_path, '*', '*.mat'))
for file in eye_files:
    mat = sio.loadmat(file)
    proc_eye = preprocess_eye(mat)
    # 저장 파일명: .npy로 저장 (dict 저장을 위해 np.savez)
    base_name = os.path.basename(file).replace('.mat', '.npz')
    save_path = os.path.join(processed_eye_path, base_name)
    np.savez(save_path, **proc_eye)
    print(f"Processed Eye data saved: {save_path}")

# ----- 시각화 예시 -----

# EEG 시각화: 전처리된 데이터 중 한 파일을 불러와 시계열 신호 일부를 plot
eeg_vis_files = glob.glob(os.path.join(processed_eeg_path, '*.npy'))
if eeg_vis_files:
    sample_eeg = np.load(eeg_vis_files[0])
    plt.figure(figsize=(12, 4))
    plt.plot(sample_eeg[:1000])  # 처음 1000 포인트
    plt.title("EEG Signal (Normalized) Sample")
    plt.xlabel("Time (samples)")
    plt.ylabel("Normalized Amplitude")
    plt.show()

# Eye data 시각화: 한 파일의 각 trial의 mean 값을 바 그래프로 시각화
eye_vis_files = glob.glob(os.path.join(processed_eye_path, '*.npz'))
if eye_vis_files:
    eye_data = np.load(eye_vis_files[0], allow_pickle=True)
    trial_keys = list(eye_data.keys())
    means = [eye_data[k].item()['mean'] for k in trial_keys]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=trial_keys, y=means)
    plt.xticks(rotation=45)
    plt.title("Eye Movement Data: Trial Means")
    plt.ylabel("Mean Value")
    plt.show()
