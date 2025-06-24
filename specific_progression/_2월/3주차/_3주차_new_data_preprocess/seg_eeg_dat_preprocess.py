import os
import pickle
import numpy as np
import scipy.signal as signal
import gc

# === 기본 설정 ===
SAMPLING_RATE = 128          # Hz
TOTAL_SAMPLES = 8064         # 원본 EEG 신호의 총 샘플 수 (63초)
BASELINE_SIZE = 3 * SAMPLING_RATE  # 3초에 해당하는 샘플 수: 384
WINDOW_SIZE = 1 * SAMPLING_RATE    # 1초 단위: 128 샘플

# 4~45Hz Bandpass 필터 설계 (Butterworth, order 4)
lowcut = 4
highcut = 45
nyquist = 0.5 * SAMPLING_RATE
b, a = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')

# === 폴더 경로 설정 ===
data_folder = "/home/bcml1/2025_EMOTION/data_preprocessed_python"  # 원본 .dat 파일 경로
output_folder = "/home/bcml1/2025_EMOTION/DEAP_EEG3"                # 결과 저장 폴더
os.makedirs(output_folder, exist_ok=True)
print(f"✅ 저장 폴더 생성 완료: {output_folder}")

# 라벨 파일이 저장된 폴더 (파일명: sXX_emotion_labels.npy)
labels_folder = "/home/bcml1/2025_EMOTION/DEAP_four_labels"

# 피험자 목록 (예: '01', '02', ..., '32')
subjects_list = ['01', '02', '03', '04', '05', '06', '07', '08', 
                 '09', '10', '11', '12', '13', '14', '15', '16', 
                 '17', '18', '19', '20', '21', '22', '23', '24', 
                 '25', '26', '27', '28', '29', '30', '31', '32']

# === 피험자별 EEG 데이터 전처리 및 각 trial 단위 파일 저장 ===
for sub in subjects_list:
    subject_id = f"s{sub}"
    file_path = os.path.join(data_folder, subject_id + ".dat")
    print(f"\n📄 처리 중: {file_path}")
    
    # 1. 데이터 로드 (.dat 파일에서 pickle로 로딩)
    with open(file_path, 'rb') as f:
        x = pickle.load(f, encoding='latin1')
    sub_data = x['data']  # shape: (40, 32, 8064) → (trial, channel, sample)
    eeg_data = sub_data[:, :32, :]    # EEG 신호만 선택 (32 채널)
    print(f"   원본 데이터 shape: {eeg_data.shape}")
    
    # 2. 맨 처음 3초(384 샘플) 제거 → 남은 샘플 수: 8064 - 384 = 7680
    eeg_data = eeg_data[:, :, BASELINE_SIZE:]
    print(f"   3초 제거 후 데이터 shape: {eeg_data.shape}")
    
    # 3. 마지막 3초(384 샘플)를 기준으로 베이스라인 평균 계산 (각 trial, 채널별)
    baseline_mean = np.mean(eeg_data[:, :, -BASELINE_SIZE:], axis=2, keepdims=True)
    # 4. 베이스라인 제거 적용 (전체 신호에서 기준값 빼기)
    eeg_data_cleaned = eeg_data - baseline_mean
    
    # 5. 4~45Hz 밴드패스 필터 적용 (각 trial, 채널별 필터링)
    eeg_data_filtered = signal.filtfilt(b, a, eeg_data_cleaned, axis=2)
    
    # 6. 각 trial을 1초(128 샘플) 단위로 분할 → 각 trial 당 7680/128 = 60 sample 생성
    num_trials, num_channels, num_samples = eeg_data_filtered.shape
    num_segments = num_samples // WINDOW_SIZE  # 7680 // 128 = 60
    print(f"   각 trial 당 {num_segments} sample 생성, 각 sample shape: ({num_channels}, {WINDOW_SIZE})")
    
    # 7. 라벨 파일 로드 (파일명: sXX_emotion_labels.npy, shape: (40,))
    label_file = os.path.join(labels_folder, f"{subject_id}_emotion_labels.npy")
    if not os.path.exists(label_file):
        print(f"   라벨 파일이 없습니다: {label_file}")
        continue
    trial_labels = np.load(label_file)  # shape: (40,)
    print(f"   라벨 파일 로드: {label_file}, shape: {trial_labels.shape}")
    
    # 8. 각 trial에 대해 60 sample을 하나의 배열로 구성하고 저장
    for trial in range(num_trials):
        trial_segments = np.zeros((num_segments, num_channels, WINDOW_SIZE))
        for seg in range(num_segments):
            start = seg * WINDOW_SIZE
            trial_segments[seg, :, :] = eeg_data_filtered[trial, :, start:start+WINDOW_SIZE]
        
        # 9. 각 trial의 sample별 Z-score 표준화 (각 sample마다)
        mean_seg = np.mean(trial_segments, axis=2, keepdims=True)
        std_seg = np.std(trial_segments, axis=2, keepdims=True)
        trial_segments_standardized = (trial_segments - mean_seg) / (std_seg + 1e-8)
        # trial_segments_standardized shape: (60, 32, 128)
        
        # 10. 파일 저장 (파일명에 trial index와 라벨 포함)
        trial_label = trial_labels[trial]
        file_name = f"{subject_id}_trial_{trial:02d}_label_{trial_label}.npy"
        save_path = os.path.join(output_folder, file_name)
        np.save(save_path, trial_segments_standardized)
        print(f"   ✅ 저장 완료: {save_path} - Shape: {trial_segments_standardized.shape}")
    
    # 메모리 정리
    del x, sub_data, eeg_data, eeg_data_cleaned, eeg_data_filtered
    gc.collect()

saved_files = [f for f in os.listdir(output_folder) if f.endswith(".npy")]
print(f"\n📂 저장된 파일 개수: {len(saved_files)}")
print(f"📄 저장된 파일 예시: {saved_files[:5]}")
