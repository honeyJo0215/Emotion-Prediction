import os
import numpy as np
import matplotlib.pyplot as plt

# 데이터 디렉토리 설정
source_dir = "~/2025_EMOTION/DEAP_EEG/ch_BPF"
source_dir = os.path.expanduser(source_dir)

# 결과 저장 디렉터리 설정
output_dir = "~/sigenv/eeg_showdata1"
output_dir = os.path.expanduser(output_dir)

# 출력 디렉터리 생성 (없을 경우)
os.makedirs(output_dir, exist_ok=True)

# .npy 파일 검색 및 처리
for file_name in os.listdir(source_dir):
    if file_name.endswith(".npy"):
        npy_path = os.path.join(source_dir, file_name)
        print(f"Processing: {npy_path}")

        # .npy 파일 로드
        data = np.load(npy_path)

        # 시각화 (전체 채널 데이터 기준)
        if len(data.shape) == 3:  # 3D 데이터만 처리
            time_steps = data.shape[1]  # 시간 축 크기
            num_channels = data.shape[2]  # 채널 수

            # 모든 채널의 데이터를 선 그래프로 시각화
            plt.figure(figsize=(15, 10))
            for channel in range(num_channels):
                plt.plot(range(time_steps), data[0, :, channel], label=f"Channel {channel}")

            plt.title(f"EEG Signal Visualization: {file_name}")
            plt.xlabel("Time Index")
            plt.ylabel("Amplitude")
            plt.legend(loc="upper right")

            # 저장 경로
            output_path = os.path.join(output_dir, file_name.replace(".npy", "_lineplot.png"))
            plt.savefig(output_path)
            plt.close()  # 플롯 닫기
            print(f"Saved line plot visualization as: {output_path}")
        else:
            print(f"Skipping {file_name}: Not 3D data.")
