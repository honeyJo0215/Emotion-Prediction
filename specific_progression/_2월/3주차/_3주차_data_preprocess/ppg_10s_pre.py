import os
import numpy as np
from scipy.signal import butter, filtfilt

# 입력 및 출력 경로 설정
input_dir = "/home/bcml1/2025_EMOTION/DEAP_npy_files+label/"
output_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_10s_real/"

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# Butterworth Bandpass 필터 정의 (0.5–5Hz)
def bandpass_filter(data, lowcut=0.5, highcut=5.0, fs=128, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=1)

# 이동 평균 필터 적용 (Smoothing, window=5)
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")

# 처리할 피험자 수 (s01 ~ s32)
for subject_id in range(1, 33):
    subject_filename = f"s{subject_id:02d}_signals.npy"
    subject_filepath = os.path.join(input_dir, subject_filename)

    if not os.path.exists(subject_filepath):
        print(f"파일을 찾을 수 없음: {subject_filepath}")
        continue

    # 데이터 로드
    subject_data = np.load(subject_filepath, allow_pickle=True)

    # 39번째 채널(PPG 신호) 추출 (데이터 크기: (40, 8064))
    ppg_data = subject_data[:, 38, :]

    # # Pre-trial Baseline Removal: 마지막 3초(384 샘플) 평균값 제거 (논문 방식)
    # baseline = np.mean(ppg_data[:, -384:], axis=1, keepdims=True)
    # ppg_data = ppg_data - baseline

    # 초기 3초(0~2초, 384 샘플) 제거 -> 03초부터 63초까지만 사용 (7680 샘플)
    ppg_data = ppg_data[:, 384:]

    # # Bandpass Filtering (0.5–5Hz)
    # ppg_data = bandpass_filter(ppg_data)

    # Moving Average Smoothing 적용
    ppg_data = np.apply_along_axis(moving_average, axis=1, arr=ppg_data)

    # # Z-score 표준화
    # mean = np.mean(ppg_data, axis=1, keepdims=True)
    # std = np.std(ppg_data, axis=1, keepdims=True)
    # ppg_data = (ppg_data - mean) / std

    # 7680 샘플을 1280 샘플씩 나눠 6개 세그먼트 생성 (각 세그먼트: 10초 길이)
    num_segments = 6
    segment_length = 1280  # 10초에 해당하는 샘플 수 (128Hz 기준)
    ppg_segments = np.split(ppg_data, num_segments, axis=1)  # 6개의 (40, 1280) 배열

    # 분할된 데이터 저장
    for seg_idx, segment in enumerate(ppg_segments):
        segment_filename = f"s{subject_id:02d}_ppg_signals_segment_{seg_idx:02d}.npy"
        segment_filepath = os.path.join(output_dir, segment_filename)
        np.save(segment_filepath, segment)

    print(f"✅ {subject_filename} 변환 완료: {len(ppg_segments)}개 세그먼트 저장됨.")

print("🎯 모든 데이터 변환 완료! 변환된 파일이 DEAP_PPG 폴더에 저장되었습니다.")
