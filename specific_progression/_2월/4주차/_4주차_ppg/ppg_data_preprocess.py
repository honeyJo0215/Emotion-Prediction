import os
import numpy as np
from scipy.signal import butter, filtfilt, resample
from scipy.signal import decimate

# 입력 및 출력 경로 설정
input_dir = "/home/bcml1/2025_EMOTION/DEAP_npy_files+label/"
label_dir = "/home/bcml1/2025_EMOTION/DEAP_four_labels/"
output_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/"
os.makedirs(output_dir, exist_ok=True)

# === Butterworth Bandpass 필터 (2차, 0.5Hz ~ 5Hz) ===
def bandpass_filter(data, lowcut=0.5, highcut=5.0, fs=128, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=1)

# === Min-Max 정규화 ([-1,1] 범위로 변환) ===
def min_max_normalization(data):
    min_val = np.min(data, axis=1, keepdims=True)
    max_val = np.max(data, axis=1, keepdims=True)
    return 2 * (data - min_val) / (max_val - min_val) - 1

# === Sliding Window 적용 (윈도우 크기: 10초, Stride: 1초) ===
def sliding_window(data, window_size=1280, stride=128):
    num_samples = data.shape[1]
    segments = []
    for start in range(0, num_samples - window_size + 1, stride):
        segments.append(data[:, start:start+window_size])
    return np.array(segments)  # 예상 Shape: (num_windows, num_trials, window_size)

# === 추가 변환 함수 ===
# 1️⃣ 이동 평균 필터 적용 (Smoothing)
def moving_average(data, window_size=5):
    return np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), axis=1, arr=data)

# 2️⃣ 다운샘플링 (Downsampling) - 논문에 맞춘 함수: 일정 간격 샘플 유지 후 보간하여 target_shape로 맞춤
def downsample(data, factor=2, target_shape=1280):
    downsampled = data[:, ::factor]  # 일정한 간격으로 샘플 유지
    return resample(downsampled, target_shape, axis=1)  # 보간하여 target_shape로 변환

# === 감정 라벨 분류 함수 ===
def get_emotion_label(valence, arousal):
    if valence > 5 and arousal > 5:
        return 0  # excited
    elif valence > 5 and arousal < 5:
        return 1  # relaxed
    elif valence < 5 and arousal > 5:
        return 2  # stressed
    else:
        return 3  # bored

# === 데이터 처리 시작 ===
for subject_id in range(1, 23):
    subject_filename = f"s{subject_id:02d}_signals.npy"
    subject_filepath = os.path.join(input_dir, subject_filename)
    label_filepath = os.path.join(label_dir, f"s{subject_id:02d}_emotion_labels.npy")

    if not os.path.exists(subject_filepath) or not os.path.exists(label_filepath):
        print(f"파일을 찾을 수 없음: {subject_filepath} 또는 {label_filepath}")
        continue

    # 1️⃣ 데이터 로드
    subject_data = np.load(subject_filepath, allow_pickle=True)

    # 2️⃣ PPG 신호 (39번 채널) 추출 → 예상 Shape: (40, 8064)
    ppg_data = subject_data[:, 38, :]

    # 3️⃣ 초기 3초(0~2초) 제거 → 예상 Shape: (40, 7680)
    ppg_data = ppg_data[:, 384:]

    # 4️⃣ Bandpass Filtering (0.5Hz ~ 5Hz)
    ppg_filtered = bandpass_filter(ppg_data)

    # 5️⃣ Min-Max 정규화
    ppg_normalized = min_max_normalization(ppg_filtered)

    # 6️⃣ 데이터 변환 (3채널 생성)
    ppg_smooth = moving_average(ppg_normalized)  # Smoothing 적용
    ppg_downsampled = downsample(ppg_normalized)   # Downsampling 적용 후 원본 크기(1280)로 변환

    # 7️⃣ Sliding Window 적용 → 각 변환별 예상 Shape: (num_windows, 40, 1280)
    segments_original = sliding_window(ppg_normalized)
    segments_smooth = sliding_window(ppg_smooth)
    segments_downsampled = sliding_window(ppg_downsampled)

    # 8️⃣ 감정 라벨 로드 (Shape: (40, 2))
    labels = np.load(label_filepath)

    # 9️⃣ Trial 별로 저장
    for trial_idx in range(40):
        # 각 trial에 대해 슬라이딩 윈도우의 두 번째 축(index 1) 선택 → Shape: (num_windows, 1280)
        trial_original = segments_original[:, trial_idx, :]
        trial_smooth = segments_smooth[:, trial_idx, :]
        trial_downsampled = segments_downsampled[:, trial_idx, :]

        # 배열들의 shape을 확인하고, 동일하지 않으면 가장 작은 shape로 자르기
        min_rows = min(trial_original.shape[0], trial_smooth.shape[0], trial_downsampled.shape[0])
        min_cols = min(trial_original.shape[1], trial_smooth.shape[1], trial_downsampled.shape[1])
        trial_original = trial_original[:min_rows, :min_cols]
        trial_smooth = trial_smooth[:min_rows, :min_cols]
        trial_downsampled = trial_downsampled[:min_rows, :min_cols]

        # 3채널 데이터를 생성하여 최종 입력으로 변환 (예상 Shape: (min_rows, 3, min_cols))
        emcnn_input = np.stack([trial_original, trial_smooth, trial_downsampled], axis=1)

        # 감정 라벨 계산
        # valence, arousal = labels[trial_idx]
        # emotion_label = get_emotion_label(valence, arousal)
        emotion_label = labels[trial_idx]
        
        # 파일명에 라벨 추가하여 저장
        trial_filename = f"s{subject_id:02d}_trial_{trial_idx:02d}_label_{emotion_label}.npy"
        trial_filepath = os.path.join(output_dir, trial_filename)
        np.save(trial_filepath, emcnn_input)

    print(f"✅ {subject_filename} 변환 완료: {40}개 trial 저장됨.")

print("🎯 모든 데이터 변환 완료! 변환된 파일이 DEAP_PPG_10s_1soverlap_4label 폴더에 저장되었습니다.")
