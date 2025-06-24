import os
import numpy as np
from scipy.signal import butter, filtfilt, resample

# 입력 및 출력 경로 설정
input_dir = "/home/bcml1/2025_EMOTION/DEAP_npy_files+label/"
label_dir = "/home/bcml1/2025_EMOTION/DEAP_four_labels/"
output_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_1s/"
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

# === 1초 단위 (128 샘플)로 Segmentation (overlap 없음) ===
def segment_by_second(data, segment_size=128):
    num_samples = data.shape[1]
    num_segments = num_samples // segment_size  # 전체 데이터에서 1초(128 샘플)씩 분할
    return np.array(np.split(data[:, :num_segments * segment_size], num_segments, axis=1))  # 결과 shape: (num_segments, num_trials, segment_size)

# === Multi-Frequency Smoothing (s=2, s=3 적용) ===
def multi_frequency_smoothing(data, scales=[2, 3]):
    smoothed_signals = []
    for s in scales:
        smoothed = np.apply_along_axis(lambda m: np.convolve(m, np.ones(s)/s, mode='same'), axis=1, arr=data)
        smoothed_signals.append(smoothed)
    return smoothed_signals  # 각 배열 shape: (num_trials, signal_length)

# === Downsampling을 각 segment에 적용하는 함수 ===
def downsample_segment(segment, d, target_length=128):
    down = segment[:, ::d]
    down_resampled = resample(down, target_length, axis=1)  # 다운샘플링 후 다시 128 길이로 맞춤
    return down_resampled

# === 감정 라벨 불러오기 ===
def load_emotion_labels(filepath):
    labels = np.load(filepath)  # Shape: (40, 2)  # valence, arousal 값
    return labels  # (40, 2)

# === 데이터 처리 시작 ===
for subject_id in range(1, 23):
    subject_filename = f"s{subject_id:02d}_signals.npy"
    subject_filepath = os.path.join(input_dir, subject_filename)
    label_filepath = os.path.join(label_dir, f"s{subject_id:02d}_emotion_labels.npy")

    if not os.path.exists(subject_filepath) or not os.path.exists(label_filepath):
        print(f"파일을 찾을 수 없음: {subject_filepath} 또는 {label_filepath}")
        continue

    # 1️⃣ 데이터 로드
    subject_data = np.load(subject_filepath, allow_pickle=True)  # Shape: (40, 8064)

    # 2️⃣ PPG 신호 (39번 채널) 추출 → 예상 Shape: (40, 8064)
    ppg_data = subject_data[:, 38, :]

    # 3️⃣ 초기 3초(0~2초) 제거 → 예상 Shape: (40, 7680)
    ppg_data = ppg_data[:, 384:]

    # 4️⃣ 샘플 수가 7680이 아니면 7680으로 보간
    if ppg_data.shape[1] != 7680:
        ppg_data = resample(ppg_data, 7680, axis=1)

    # 5️⃣ Bandpass Filtering (0.5Hz ~ 5Hz)
    ppg_filtered = bandpass_filter(ppg_data)

    # 6️⃣ Min-Max 정규화
    ppg_normalized = min_max_normalization(ppg_filtered)

    # 7️⃣ Identity Mapping (원본 데이터 유지)
    ppg_identity = ppg_normalized.copy()

    # 8️⃣ Multi-Frequency 변환 (다양한 스케일에서 smoothing)
    ppg_smooth_s2, ppg_smooth_s3 = multi_frequency_smoothing(ppg_normalized)

    # 9️⃣ 1초 단위로 Segmentation 적용 (overlap 없음)
    segments_identity = segment_by_second(ppg_identity)  # Shape: (num_segments, 40, 128)
    segments_smooth_s2 = segment_by_second(ppg_smooth_s2)
    segments_smooth_s3 = segment_by_second(ppg_smooth_s3)

    # 🔟 Downsampling 적용 (d=2, d=3)
    segments_downsampled_d2 = np.array([downsample_segment(seg, d=2) for seg in segments_identity])  # shape: (num_segments, 40, 128)
    segments_downsampled_d3 = np.array([downsample_segment(seg, d=3) for seg in segments_identity])

    # 1️⃣1️⃣ 감정 라벨 불러오기 (Shape: (40, 2))
    labels = load_emotion_labels(label_filepath)

    # 1️⃣2️⃣ Trial 별로 최종 데이터를 생성 및 저장
    for trial_idx in range(40):
        trial_identity = segments_identity[:, trial_idx, :]       # shape: (num_segments, 128)
        trial_smooth_s2 = segments_smooth_s2[:, trial_idx, :]     # shape: (num_segments, 128)
        trial_smooth_s3 = segments_smooth_s3[:, trial_idx, :]     # shape: (num_segments, 128)
        trial_downsampled_d2 = segments_downsampled_d2[:, trial_idx, :]  # shape: (num_segments, 128)
        trial_downsampled_d3 = segments_downsampled_d3[:, trial_idx, :]  # shape: (num_segments, 128)

        # 5채널 데이터를 스택하여 최종 입력으로 변환: (num_segments, 5, 128)
        emcnn_input = np.stack([trial_identity, trial_smooth_s2, trial_smooth_s3, 
                                trial_downsampled_d2, trial_downsampled_d3], axis=1)

        # 감정 라벨 적용 및 저장
        emotion_label = labels[trial_idx]
        trial_filename = f"s{subject_id:02d}_trial_{trial_idx:02d}_label_{emotion_label}.npy"
        trial_filepath = os.path.join(output_dir, trial_filename)
        np.save(trial_filepath, emcnn_input)

    print(f"✅ {subject_filename} 변환 완료: {40}개 trial 저장됨.")

print("🎯 모든 데이터 변환 완료! 변환된 파일이 DEAP_PPG_1s 폴더에 저장되었습니다.")
