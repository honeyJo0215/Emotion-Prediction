# 수정중

import os
import numpy as np

# 📌 원본 EEG NPY 파일 폴더 및 저장 폴더 설정
eeg_npy_folder = "/home/bcml1/2025_EMOTION/DEAP_eeg_npy_files"
output_folder = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap/"

# 저장 폴더 생성 확인
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print(f"✅ 저장 폴더 생성 완료: {output_folder}")

# === 기본 설정 ===
SAMPLING_RATE = 128
TOTAL_SAMPLES = 8064  # 총 샘플 수
BASELINE_SIZE = 3 * SAMPLING_RATE  # 마지막 3초 (384 샘플)
WINDOW_SIZE = 10 * SAMPLING_RATE  # 10초 (1280 샘플)
STEP_SIZE = 1 * SAMPLING_RATE  # 1초 overlap (128 샘플)

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

    # === 2.2 마지막 3초(384 샘플) 기준으로 베이스라인 평균 계산 ===
    baseline_mean = np.mean(eeg_data[:, :, -BASELINE_SIZE:], axis=2, keepdims=True)  # Shape: (40, 32, 1)

    # === 2.3 베이스라인 제거 적용 ===
    eeg_data_cleaned = eeg_data[:, :, :-BASELINE_SIZE] - baseline_mean  # (40, 32, 7680)

    # === 2.4 10초 단위 윈도우 분할 (1초씩 overlap) ===
    num_samples = num_samples - BASELINE_SIZE  # 8064 → 7680 샘플

    for trial in range(num_trials):  # 각 sample(trial) 별로 처리
        segment_idx = 1  # 🎯 세그먼트 번호를 샘플별로 001부터 초기화

        for start in range(0, num_samples - WINDOW_SIZE + 1, STEP_SIZE):  # 1초씩 이동
            segment = eeg_data_cleaned[trial, :, start:start + WINDOW_SIZE]  # (채널 수, 1280)

            # === 2.5 Z-score 표준화 (확실한 표준화) ===
            mean = np.mean(segment, axis=1, keepdims=True)
            std = np.std(segment, axis=1, keepdims=True)
            segment = (segment - mean) / (std + 1e-8)  # 0으로 나누는 문제 방지

            # === 2.6 저장 ===
            segment_filename = f"{subject_id}_sample_{trial+1:02d}_segment_{segment_idx:03d}.npy"
            save_path = os.path.join(output_folder, segment_filename)
            np.save(save_path, segment)

            # 저장 확인
            if os.path.exists(save_path):
                print(f"✅ 저장 완료: {segment_filename}")
            else:
                print(f"❌ 저장 실패: {segment_filename}")

            segment_idx += 1  # 🎯 세그먼트 번호 증가

# 저장된 파일 개수 확인
output_files = [f for f in os.listdir(output_folder) if f.endswith(".npy")]
print(f"📂 저장된 파일 개수: {len(output_files)}")
print(f"📄 저장된 파일 예시: {output_files[:5]}")
