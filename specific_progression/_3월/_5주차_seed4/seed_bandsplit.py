import numpy as np
import os
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    지정한 lowcut ~ highcut 대역으로 Butterworth 밴드패스 필터를 적용합니다.
    data: 입력 신호 (채널별, n_samples)
    fs: 샘플링 주파수 (Hz)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data, axis=-1)
    return filtered

def main():
    # 경로 설정
    sample_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/1_npy_sample"
    save_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/1_npy_band"
    os.makedirs(save_dir, exist_ok=True)
    
    # npy 파일 목록 가져오기
    npy_files = [f for f in os.listdir(sample_dir) if f.endswith('.npy')]
    
    fs = 200  # 다운샘플링 후 주파수 (Hz)
    
    # 각 주파수 대역 (알파, 베타, 감마, 세타)
    bands = {
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 45),
        'theta': (4, 8)
    }
    
    for file_name in npy_files:
        file_path = os.path.join(sample_dir, file_name)
        data = np.load(file_path)  # data.shape: (62, n_samples)
        n_channels, n_samples = data.shape
        
        # 각 대역 필터링 결과를 저장할 배열 (62, n_samples, 4)
        filtered_data = np.zeros((n_channels, n_samples, 4))
        
        # 순서: 알파, 베타, 감마, 세타
        for idx, band in enumerate(['alpha', 'beta', 'gamma', 'theta']):
            low, high = bands[band]
            filtered = bandpass_filter(data, low, high, fs, order=4)
            filtered_data[:, :, idx] = filtered
        
        # 결과 저장 (동일 파일명으로 저장)
        save_path = os.path.join(save_dir, file_name)
        np.save(save_path, filtered_data)
        print(f"Saved filtered data to {save_path}")

if __name__ == '__main__':
    main()
