#예상결과 shape: (40, 32, 128)
# 결과 파일 이름 형식: sXX_eeg_signals_segment_XX.npy
import os
import numpy as np
import scipy.signal as signal

# 📌 원본 EEG NPY 파일 폴더 및 저장 폴더 설정
eeg_npy_folder = "/home/bcml1/2025_EMOTION/DEAP_eeg_npy_files"
output_folder = "/home/bcml1/2025_EMOTION/DEAP_EEG/"

# 저장 폴더 생성 확인
os.makedirs(output_folder, exist_ok=True)
print(f"✅ 저장 폴더 생성 완료: {output_folder}")

# === 기본 설정 ===
SAMPLING_RATE = 128  # Hz
TOTAL_SAMPLES = 8064  # 총 샘플 수
BASELINE_SIZE = 3 * SAMPLING_RATE  # 384 샘플 (마지막 3초)
WINDOW_SIZE = 1 * SAMPLING_RATE  # 128 샘플 (1초 단위)

# 4~45Hz Bandpass 필터 설계
lowcut = 4
highcut = 45
nyquist = 0.5 * SAMPLING_RATE
b, a = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')

# === 1. 파일 목록 가져오기 ===
eeg_files = sorted([f for f in os.listdir(eeg_npy_folder) if f.endswith("_signals.npy")])
print(f"📂 총 {len(eeg_files)}개의 EEG 파일을 찾았습니다.")

# === 2. 파일별 전처리 ===
for eeg_file in eeg_files:
    file_path = os.path.join(eeg_npy_folder, eeg_file)
    
    # === 2.1 데이터 로드 ===
    eeg_data = np.load(file_path)  # Shape: (40, 32, 8064)
    subject_id = eeg_file.split("_")[0]  # sXX 추출
    print(f"📄 처리 중: {eeg_file} - 원본 데이터 형태: {eeg_data.shape}")

    num_trials, num_channels, num_samples = eeg_data.shape  # (40, 32, 8064)

    # === 2.2 맨 처음 3초(384 샘플) 제거 ===
    eeg_data = eeg_data[:, :, BASELINE_SIZE:]  # (40, 32, 7680)
    num_samples = num_samples - BASELINE_SIZE  # 8064 → 7680 샘플

    # === 2.3 마지막 3초(384 샘플) 기준으로 베이스라인 평균 계산 ===
    baseline_mean = np.mean(eeg_data[:, :, -BASELINE_SIZE:], axis=2, keepdims=True)  # Shape: (40, 32, 1)
    
    # === 2.4 베이스라인 제거 적용 (마지막 3초도 포함) ===
    eeg_data_cleaned = eeg_data - baseline_mean  # (40, 32, 7680)
    
    # === 2.5 4~45Hz Bandpass 필터 적용 ===
    eeg_data_filtered = signal.filtfilt(b, a, eeg_data_cleaned, axis=2)
    
    # === 2.6 1초 단위로 데이터 분할 ===
    num_segments = num_samples // WINDOW_SIZE  # 7680 // 128 = 60
    eeg_segments = np.zeros((num_trials, num_channels, num_segments, WINDOW_SIZE))  # (40, 32, 60, 128)

    for segment_idx in range(num_segments):
        start = segment_idx * WINDOW_SIZE
        eeg_segments[:, :, segment_idx, :] = eeg_data_filtered[:, :, start:start + WINDOW_SIZE]  # (40, 32, 128)
    
    # === 2.7 Z-score 표준화 (각 segment 별로) ===
    mean = np.mean(eeg_segments, axis=(3), keepdims=True)  # (40, 32, 60, 1)
    std = np.std(eeg_segments, axis=(3), keepdims=True)  # (40, 32, 60, 1)
    eeg_segments_standardized = (eeg_segments - mean) / (std + 1e-8)  # (40, 32, 60, 128)

    # === 2.8 저장 (각 segment 별로 저장) ===
    for segment_idx in range(num_segments):
        segment_filename = f"{subject_id}_eeg_signals_segment_{segment_idx:02d}.npy"
        save_path = os.path.join(output_folder, segment_filename)
        np.save(save_path, eeg_segments_standardized[:, :, segment_idx, :])  # (40, 32, 128)
        print(f"✅ 저장 완료: {save_path} - Shape: {eeg_segments_standardized[:, :, segment_idx, :].shape}")

# 저장된 파일 개수 확인
output_files = [f for f in os.listdir(output_folder) if f.endswith(".npy")]
print(f"📂 저장된 파일 개수: {len(output_files)}")
print(f"📄 저장된 파일 예시: {output_files[:5]}")
