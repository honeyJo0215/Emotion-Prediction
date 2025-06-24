import os
import numpy as np
import scipy.signal as signal

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    주어진 1차원 데이터에 대해 butterworth bandpass filter를 적용.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def extract_de_features(eeg_data, fs=200):
    """
    eeg_data: numpy array, shape = (channels, total_samples)
    fs: 목표 샘플링 주파수 (Hz) -> 200 Hz
    반환: de_features, shape = (channels, n_segments, 4)
    """
    channels, total_samples = eeg_data.shape
    # 1초(200 샘플) 단위로 분할
    segment_length = fs  
    n_segments = total_samples // segment_length  # 정수 몫
    
    # 결과를 저장할 배열 초기화
    de_features = np.zeros((channels, n_segments, 4))
    
    # 사용 주파수 대역 (예시: delta, theta, alpha, beta)
    bands = [(1, 4), (4, 8), (8, 14), (14, 30)]
    
    # 각 채널에 대해 처리
    for ch in range(channels):
        # 각 주파수 대역별로 filtering 후 세그먼트별로 DE feature 계산
        for band_idx, (lowcut, highcut) in enumerate(bands):
            # 해당 채널에 대해 bandpass filtering
            filtered_signal = bandpass_filter(eeg_data[ch, :], lowcut, highcut, fs)
            # 1초(200 샘플) 단위로 분할하여 로그 분산(DE) 계산
            for seg in range(n_segments):
                seg_start = seg * segment_length
                seg_end = seg_start + segment_length
                segment = filtered_signal[seg_start:seg_end]
                # 분산 계산 후 로그 취함 (0에 대한 대비로 작은 epsilon 추가)
                var_seg = np.var(segment)
                de_features[ch, seg, band_idx] = np.log(var_seg + 1e-8)
    
    return de_features

def main():
    # 입력 및 출력 경로 설정
    input_dir = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/3_npy_sample'
    output_dir = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE/3'
    os.makedirs(output_dir, exist_ok=True)
    
    # 폴더 내의 모든 npy 파일 리스트업
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    for fname in npy_files:
        file_path = os.path.join(input_dir, fname)
        # npy 파일 불러오기 (형태: (62, X))
        eeg_data = np.load(file_path)
        
        # 'XXXXX에서 1을 빼서' -> 시간 축에서 마지막 샘플 제거 (예: (62, X-1))
        eeg_data = eeg_data[:, :-1]
        
        # 만약 원본 데이터가 200Hz가 아니라면 아래와 같이 resample 가능
        # 예를 들어, original_fs가 주어졌다면:
        # original_fs = 256  # 예시
        # n_samples_new = int(eeg_data.shape[1] / original_fs * 200)
        # eeg_data = signal.resample(eeg_data, n_samples_new, axis=1)
        
        # DE feature 추출 (각 1초당 200 샘플)
        de_features = extract_de_features(eeg_data, fs=200)
        
        # 결과 저장 (같은 파일명으로 저장)
        output_file = os.path.join(output_dir, fname)
        np.save(output_file, de_features)
        print(f"Processed and saved: {output_file}")

if __name__ == '__main__':
    main()
