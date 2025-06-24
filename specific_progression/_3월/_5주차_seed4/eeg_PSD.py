import numpy as np
import os

def extract_psd_feature(data, fs=200):
    """
    data: (n_channels, n_samples, n_bands)
    fs: 샘플링 주파수 (Hz) - 여기서는 200Hz
    각 채널/밴드별로 첫 1 샘플을 제거한 후, 남은 데이터를 1초(200 샘플) 단위로 구분하여
    제곱 평균을 계산합니다.
    
    반환값: (n_channels, n_segments, n_bands)
    """
    n_channels, n_samples, n_bands = data.shape
    
    # 잡신호 1 제거: n_samples - 1 이 200으로 나누어 떨어짐
    trimmed_data = data[:, 1:, :]  # (n_channels, n_samples-1, n_bands)
    total_samples = trimmed_data.shape[1]
    n_segments = total_samples // fs  # 1초에 fs 샘플
    
    # 1초 단위로 reshape: (n_channels, n_segments, fs, n_bands)
    reshaped = trimmed_data[:, :n_segments * fs, :].reshape(n_channels, n_segments, fs, n_bands)
    
    # 각 1초 구간의 PSD feature는 제곱 평균 (power)로 계산
    psd_feature = np.mean(reshaped ** 2, axis=2)  # (n_channels, n_segments, n_bands)
    return psd_feature

def main():
    # 입력 및 출력 경로 설정
    band_data_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/3_npy_band"
    psd_save_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_PSD/3"
    os.makedirs(psd_save_dir, exist_ok=True)
    
    # 밴드 데이터 파일 목록 가져오기 (.npy 파일)
    band_files = [f for f in os.listdir(band_data_dir) if f.endswith('.npy')]
    
    for file_name in band_files:
        file_path = os.path.join(band_data_dir, file_name)
        data = np.load(file_path)  # data.shape: (62, n_samples, 4)
        
        # PSD feature 추출
        psd = extract_psd_feature(data, fs=200)
        # psd.shape은 예를 들어 (62, 160, 4)가 됨
        
        # 결과 저장 (동일 파일명으로 저장)
        save_path = os.path.join(psd_save_dir, file_name)
        np.save(save_path, psd)
        print(f"Saved PSD features to {save_path}")

if __name__ == '__main__':
    main()
