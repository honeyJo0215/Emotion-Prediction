import os
import numpy as np
import matplotlib.pyplot as plt

# EEG 데이터 로드 (파일 경로를 환경에 맞게 수정)
eeg_data = np.load('/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/1_npy_sample/1_20160518_sample_01_label_1.npy')
n_channels, n_samples = eeg_data.shape

# 샘플링 주파수 (Hz)
fs = 200

# 시간축 생성
time = np.arange(n_samples) / fs

# 각 채널의 데이터를 겹치지 않도록 오프셋 적용
offset = np.max(np.abs(eeg_data)) * 1.5

plt.figure(figsize=(15, 10))
for i in range(n_channels):
    plt.plot(time, eeg_data[i, :] + i * offset, label=f'Channel {i+1}')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude + offset')
plt.title('EEG Waveforms (62 channels)')
plt.tight_layout()

# 출력 디렉토리 및 파일 경로 지정
output_dir = '/home/bcml1/sigenv/_4주차_seed4/eeg_visualize'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'eeg_waveforms.png')

# PNG 파일로 저장 (해상도 dpi: 300)
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Saved PNG file to: {output_file}")
