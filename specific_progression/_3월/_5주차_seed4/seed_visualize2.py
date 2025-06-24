import os
import numpy as np
import matplotlib.pyplot as plt

# EEG 데이터 로드 (파일 경로를 환경에 맞게 수정)
# 파일의 shape은 (62, XXXXX)이며, 여기서는 numpy 배열로 저장되어 있다고 가정합니다.
eeg_data = np.load('/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/1_npy_sample/1_20160518_sample_01_label_1.npy')
n_channels, n_samples = eeg_data.shape

# 샘플링 주파수 (Hz)
fs = 800

# 추출할 5초 길이의 데이터 샘플 수 계산
segment_duration = 5  # 초 단위
segment_samples = int(fs * segment_duration)

# 데이터 길이가 5초보다 짧은 경우 전체 데이터를 사용하도록 조정
if segment_samples > n_samples:
    segment_samples = n_samples

# 5초 데이터 추출 (여기서는 시작 부분 5초 데이터를 사용)
eeg_segment = eeg_data[:, :segment_samples]

# 추출한 데이터에 맞춘 시간축 생성
time_segment = np.arange(segment_samples) / fs

# 각 채널의 파형이 겹치지 않도록 오프셋 계산 (추출한 segment 기준)
offset = np.max(np.abs(eeg_segment)) * 1.5

plt.figure(figsize=(15, 10))
for i in range(n_channels):
    plt.plot(time_segment, eeg_segment[i, :] + i * offset, label=f'Channel {i+1}')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude + offset')
plt.title(f'EEG Waveforms (62 channels) - {segment_duration} seconds segment')
plt.tight_layout()

# 출력 디렉토리 지정 및 디렉토리가 없으면 생성
output_dir = '/home/bcml1/sigenv/_4주차_seed4/eeg_visualize'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'eeg_waveforms_zoomed.png')

# PNG 파일로 저장 (dpi: 300)
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Saved PNG file to: {output_file}")
