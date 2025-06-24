#밴드패스필터와 로우패스필터 적용

import os
import pickle
import numpy as np
import scipy.signal as signal
import gc

# === 기본 설정 ===
SAMPLING_RATE = 128          # Hz
TOTAL_SAMPLES = 8064         # 원본 PPS 신호의 총 샘플 수 (63초)
BASELINE_SIZE = 3 * SAMPLING_RATE  # 3초에 해당하는 샘플 수: 384
WINDOW_SIZE = 1 * SAMPLING_RATE    # 1초 단위: 128 샘플

# 4~45Hz Bandpass 필터 설계 (Butterworth, order 4)
# lowcut = 4
# highcut = 45
# nyquist = 0.5 * SAMPLING_RATE
# b, a = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')

# === 필터 설계 ===
def bandpass_filter(data, lowcut, highcut, fs=128, order=4):
    nyquist = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)

def lowpass_filter(data, cutoff, fs=128, order=4):
    nyquist = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyquist, btype='low')
    return signal.filtfilt(b, a, data, axis=-1)

# === 폴더 경로 설정 ===
# 원본 .dat 파일 경로
data_folder = "/home/bcml1/2025_EMOTION/data_preprocessed_python"
# 결과 저장 폴더
output_folder = "/home/bcml1/2025_EMOTION/DEAP_PPS2"
os.makedirs(output_folder, exist_ok=True)
print(f"✅ 저장 폴더 생성 완료: {output_folder}")

# 피험자 목록 (예: '01', '02', ..., '32')
subjects_list = ['01', '02', '03', '04', '05', '06', '07', '08', 
                 '09', '10', '11', '12', '13', '14', '15', '16', 
                 '17', '18', '19', '20', '21', '22', '23', '24', 
                 '25', '26', '27', '28', '29', '30', '31', '32']

# === 피험자별 PPS 데이터 전처리 및 세그먼트 저장 ===
for sub in subjects_list:
    subject_id = f"s{sub}"
    file_path = os.path.join(data_folder, subject_id + ".dat")
    print(f"\n📄 처리 중: {file_path}")
    
    # 1. 데이터 로드 (.dat 파일에서 pickle 로딩)
    with open(file_path, 'rb') as f:
        x = pickle.load(f, encoding='latin1')
    sub_data = x['data']            # 보통 shape: (40, 32, 8064) → (trial, channel, sample)
    pps_data = sub_data[:, 32:40, :]  # 33번째(인덱스 32)부터 40번째(인덱스 39)까지의 채널 선택
    print(f"   원본 데이터 shape: {pps_data.shape}")
    
    # 2. 맨 처음 3초(384 샘플) 제거 → 남은 샘플 수: 8064 - 384 = 7680
    pps_data = pps_data[:, :, BASELINE_SIZE:]
    print(f"   3초 제거 후 데이터 shape: {pps_data.shape}")
    
    # 3. 마지막 3초(384 샘플)를 기준으로 베이스라인 평균 계산
    baseline_mean = np.mean(pps_data[:, :, -BASELINE_SIZE:], axis=2, keepdims=True)
    # 4. 베이스라인 제거 적용 (전체 신호에서 기준값 빼기)
    pps_data_cleaned = pps_data - baseline_mean
    
    # ✅ 원본 유지 후 필터링 적용할 새로운 변수 생성
    pps_data_filtered = np.copy(pps_data_cleaned)
    
    # === 5-1. EOG(33~34채널) Bandpass 필터 적용 (0.1~10Hz) ===
    pps_data_filtered[:, 0:2, :] = bandpass_filter(pps_data_filtered[:, 0:2, :], 0.1, 10)

    # === 5-2. EMG(35~38채널) Bandpass 필터 적용 (20~250Hz) + 절대값 변환 + Lowpass (10Hz) ===
    # pps_data_filtered[:, 2:6, :] = bandpass_filter(pps_data_filtered[:, 2:6, :], 20, 250)
    # pps_data_filtered[:, 2:6, :] = np.abs(pps_data_filtered[:, 2:6, :])
    # pps_data_filtered[:, 2:6, :] = lowpass_filter(pps_data_filtered[:, 2:6, :], 10)
    pps_data_filtered[:, 2:6, :] = np.abs(pps_data_filtered[:, 2:6, :])  # 먼저 절대값 변환
    pps_data_filtered[:, 2:6, :] = lowpass_filter(pps_data_filtered[:, 2:6, :], 10)  # 그 후 Lowpass 필터 적용

    # === 5-3. GSR(39~40채널) Lowpass 필터 적용 (0.05~5Hz) ===
    # pps_data_filtered[:, 6:8, :] = lowpass_filter(pps_data_filtered[:, 6:8, :], 5)
    pps_data_filtered[:, 6:8, :] = bandpass_filter(pps_data_filtered[:, 6:8, :], 0.05, 5)

    
    # 6. 1초 단위로 데이터 분할  
    #    전체 샘플 수: 7680 → 1초(128 샘플) 단위이면 총 7680/128 = 60 segment
    num_trials, num_channels, num_samples = pps_data_filtered.shape
    num_segments = num_samples // WINDOW_SIZE  # 7680 // 128 = 60
    pps_segments = np.zeros((num_trials, num_channels, num_segments, WINDOW_SIZE))
    
    for segment_idx in range(num_segments):
        start = segment_idx * WINDOW_SIZE
        pps_segments[:, :, segment_idx, :] = pps_data_filtered[:, :, start:start + WINDOW_SIZE]
    print(f"   데이터 분할: {num_segments} segment 생성, 각 segment shape: ({num_trials}, {num_channels}, {WINDOW_SIZE})")
    
    # 7. 각 segment별 Z-score 표준화  
    #    각 segment에 대해 (신호 - 평균) / (표준편차) 계산 (분모 0 방지를 위해 1e-8 추가)
    mean_seg = np.mean(pps_segments, axis=3, keepdims=True)
    std_seg = np.std(pps_segments, axis=3, keepdims=True)
    pps_segments_standardized = (pps_segments - mean_seg) / (std_seg + 1e-8)
    
    # 8. 각 segment별로 저장 (파일 이름 형식: sXX_pps_signals_segment_XX.npy)
    for segment_idx in range(num_segments):
        segment_filename = f"{subject_id}_pps_signals_segment_{segment_idx:02d}.npy"
        save_path = os.path.join(output_folder, segment_filename)
        # 저장되는 데이터 shape: (trial, channel, WINDOW_SIZE) 즉 (40, 32, 128)
        np.save(save_path, pps_segments_standardized[:, :, segment_idx, :])
        print(f"   ✅ 저장 완료: {save_path} - Shape: {pps_segments_standardized[:, :, segment_idx, :].shape}")
    
    # 메모리 정리
    del x, sub_data, pps_data, pps_data_cleaned, pps_data_filtered, pps_segments, pps_segments_standardized
    gc.collect()

# 저장된 파일 개수 및 일부 파일명 출력
saved_files = [f for f in os.listdir(output_folder) if f.endswith(".npy")]
print(f"\n📂 저장된 파일 개수: {len(saved_files)}")
print(f"📄 저장된 파일 예시: {saved_files[:5]}")
