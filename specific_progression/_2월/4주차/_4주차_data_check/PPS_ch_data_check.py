import os
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

# 데이터가 있는 경로와 결과 저장 경로
data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPS3'
save_dir = '/home/bcml1/sigenv/_4주차_data_check/DEAP_PPS_ch_Visualize'
os.makedirs(save_dir, exist_ok=True)

# 서브젝트 목록 (s01 ~ s22)
subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
num_trials = 40          # trial_00 ~ trial_39
num_channels = 8         # 채널 33 ~ 40

# 각 서브젝트에 대해 처리
for subject in subjects:
    # 각 채널별로 시각화 (총 8개 이미지)
    for ch in range(num_channels):
        # subplot grid: 7열로 trial 수에 맞춰 행 계산
        cols = 7
        rows = math.ceil(num_trials / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), sharex=True)
        axes = axes.flatten()

        # 각 trial에 대해 해당 채널 신호를 플롯
        for trial in range(num_trials):
            ax = axes[trial]
            # 파일명 패턴: sXX_trial_YY_label_*.npy (라벨은 인코딩되어 있음)
            pattern = os.path.join(data_dir, f"{subject}_trial_{trial:02d}_label_*.npy")
            matching_files = glob.glob(pattern)
            if not matching_files:
                print(f"File not found for {subject} trial {trial:02d}")
                continue

            # 여러 파일이 있으면 첫 번째 파일 사용
            file_path = matching_files[0]
            data = np.load(file_path)

            # 데이터 shape에 따른 처리
            if data.ndim == 2 and data.shape == (num_channels, 128):
                # (8, 128)인 경우 해당 채널 데이터 선택
                channel_data = data[ch, :]
                time_vector = np.linspace(0, 60, 128)
            elif data.ndim == 3 and data.shape == (60, num_channels, 128):
                # (60, 8, 128)인 경우 세그먼트를 연결하여 하나의 신호로 만듦
                processed_data = data.transpose(1, 0, 2).reshape(num_channels, -1)
                channel_data = processed_data[ch, :]
                time_vector = np.linspace(0, 60, processed_data.shape[1])
            else:
                print(f"Unexpected shape in {file_path}: {data.shape}")
                continue

            ax.plot(time_vector, channel_data, linewidth=1)
            ax.set_title(f"Trial {trial:02d}", fontsize=10)
            if trial % cols == 0:
                ax.set_ylabel("Signal", fontsize=8)
            if trial >= num_trials - cols:
                ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=8)
        
        # 사용하지 않는 subplot 삭제
        for i in range(num_trials, len(axes)):
            fig.delaxes(axes[i])
        
        fig.suptitle(f"Subject {subject} - Channel {ch+33} Visualization", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"{subject}_channel_{ch+33}_trials_visualization.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved visualization for {subject} channel {ch+33} at {save_path}")
