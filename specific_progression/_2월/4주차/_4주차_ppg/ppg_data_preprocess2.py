import os
import numpy as np
from scipy.signal import butter, filtfilt, resample
import glob

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

# === Sliding Window 적용 (윈도우 크기: 10초 = 1280 샘플, Stride: 1초 = 128 샘플) ===
def sliding_window(data, window_size=1280, stride=128):
    num_samples = data.shape[1]
    segments = []
    for start in range(0, num_samples - window_size + 1, stride):
        segments.append(data[:, start:start+window_size])
    return np.array(segments)  # 결과 shape: (num_windows, num_trials, window_size)

# === Multi-Frequency Smoothing (s=2, s=3 적용) ===
def multi_frequency_smoothing(data, scales=[2, 3]):
    smoothed_signals = []
    for s in scales:
        # data shape: (num_trials, signal_length)
        smoothed = np.apply_along_axis(lambda m: np.convolve(m, np.ones(s)/s, mode='same'), axis=1, arr=data)
        smoothed_signals.append(smoothed)
    return smoothed_signals  # 각 배열 shape: (num_trials, signal_length)

# === Downsampling을 각 segment에 적용하는 함수 ===
def downsample_segment(segment, d, target_length=1280):
    # segment shape: (num_trials, window_size)  window_size = 1280
    # 다운샘플링: 매 d번째 샘플 추출 → shape: (num_trials, ceil(1280/d))
    down = segment[:, ::d]
    # 보간하여 target_length (1280)로 변환
    down_resampled = resample(down, target_length, axis=1)
    return down_resampled

# === 감정 라벨 분류 함수 (예시) ===
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
    
    # 만약 샘플 수가 7680이 아니면 7680으로 보간
    if ppg_data.shape[1] != 7680:
        ppg_data = resample(ppg_data, 7680, axis=1)

    # 4️⃣ Bandpass Filtering (0.5Hz ~ 5Hz)
    ppg_filtered = bandpass_filter(ppg_data)

    # 5️⃣ Min-Max 정규화
    ppg_normalized = min_max_normalization(ppg_filtered)

    # 6️⃣ Identity Mapping (원본 데이터 유지)
    ppg_identity = ppg_normalized.copy()

    # 7️⃣ Multi-Frequency 변환 (다양한 스케일에서 smoothing)
    ppg_smooth_s2, ppg_smooth_s3 = multi_frequency_smoothing(ppg_normalized)

    # 8️⃣ Sliding Window 적용 (각 branch에 대해 적용)
    # 각 결과 shape: (51, 40, 1280) → 51 윈도우, 40 trial, 1280 샘플
    segments_identity = sliding_window(ppg_identity)
    segments_smooth_s2 = sliding_window(ppg_smooth_s2)
    segments_smooth_s3 = sliding_window(ppg_smooth_s3)

    # 9️⃣ 각 슬라이딩 윈도우에 대해 Downsampling branch 적용 (d=2, d=3)
    segments_downsampled_d2 = []
    segments_downsampled_d3 = []
    for seg in segments_identity:  # seg shape: (40, 1280)
        down_d2 = downsample_segment(seg, d=2, target_length=1280)  # shape: (40, 1280)
        down_d3 = downsample_segment(seg, d=3, target_length=1280)  # shape: (40, 1280)
        segments_downsampled_d2.append(down_d2)
        segments_downsampled_d3.append(down_d3)
    segments_downsampled_d2 = np.array(segments_downsampled_d2)  # shape: (51, 40, 1280)
    segments_downsampled_d3 = np.array(segments_downsampled_d3)  # shape: (51, 40, 1280)

    # 🔟 감정 라벨 불러오기 (Shape: (40, 2)) → 각 trial별 라벨 (예: valence, arousal)
    labels = np.load(label_filepath)

    # 1️⃣1️⃣ Trial 별로 최종 데이터를 생성 및 저장
    # 각 trial에 대해, 각 branch에서 sliding window로 자른 데이터를 추출하여,
    # 최종적으로 5채널 데이터를 스택하여 (51, 5, 1280)의 shape 생성
    for trial_idx in range(40):
        trial_identity = segments_identity[:, trial_idx, :]       # shape: (51, 1280)
        trial_smooth_s2 = segments_smooth_s2[:, trial_idx, :]       # shape: (51, 1280)
        trial_smooth_s3 = segments_smooth_s3[:, trial_idx, :]       # shape: (51, 1280)
        trial_downsampled_d2 = segments_downsampled_d2[:, trial_idx, :]  # shape: (51, 1280)
        trial_downsampled_d3 = segments_downsampled_d3[:, trial_idx, :]  # shape: (51, 1280)

        # 5채널 데이터를 스택하여 최종 입력으로 변환: (51, 5, 1280)
        emcnn_input = np.stack([trial_identity, trial_smooth_s2, trial_smooth_s3, 
                                trial_downsampled_d2, trial_downsampled_d3], axis=1)

        # 감정 라벨 (여기서는 단순히 labels[trial_idx] 사용)
        # (실제 감정 분류 기준에 따라 변환할 수 있음)
        emotion_label = labels[trial_idx]
        
        # 파일명에 라벨 추가하여 저장
        trial_filename = f"s{subject_id:02d}_trial_{trial_idx:02d}_label_{emotion_label}.npy"
        trial_filepath = os.path.join(output_dir, trial_filename)
        np.save(trial_filepath, emcnn_input)

    print(f"✅ {subject_filename} 변환 완료: {40}개 trial 저장됨.")

print("🎯 모든 데이터 변환 완료! 변환된 파일이 DEAP_PPG_10s_1soverlap_4label 폴더에 저장되었습니다.")
