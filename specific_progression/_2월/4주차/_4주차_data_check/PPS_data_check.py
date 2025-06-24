import os
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

# 데이터 경로와 결과 저장 경로
data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPS3'
save_dir = '/home/bcml1/sigenv/_4주차_data_check/DEAP_PPS_Visualize'
os.makedirs(save_dir, exist_ok=True)

# 서브젝트 목록: s01 ~ s22
subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
num_trials = 40          # trial_00 ~ trial_39
num_channels = 8         # 채널 33~40

# 각 서브젝트별로 처리
for subject in subjects:
    cols = 7
    rows = math.ceil(num_trials / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), sharex=True)
    axes = axes.flatten()
    
    for trial in range(num_trials):
        ax = axes[trial]
        # 파일명 패턴: sXX_trial_YY_label_*.npy (라벨은 파일명에 인코딩되어 있음)
        pattern = os.path.join(data_dir, f"{subject}_trial_{trial:02d}_label_*.npy")
        matching_files = glob.glob(pattern)
        if not matching_files:
            print(f"File not found for {subject} trial {trial:02d}")
            continue
        
        # 해당 trial의 파일이 여러 개일 경우 첫 번째 파일을 사용
        file_path = matching_files[0]
        data = np.load(file_path)
        
        # 두 가지 shape 모두 처리: (8, 128) 또는 (60, 8, 128)
        if data.ndim == 2 and data.shape == (num_channels, 128):
            # 이미 (8, 128) 형태인 경우
            processed_data = data
            time_vector = np.linspace(0, 60, 128)
        elif data.ndim == 3 and data.shape == (60, num_channels, 128):
            # (60, 8, 128) 형태인 경우: 60세그먼트를 하나로 이어 붙임
            # 먼저 (60, 8, 128) -> (8, 60, 128)로 축 순서를 변경한 후 reshape
            processed_data = data.transpose(1, 0, 2).reshape(num_channels, -1)
            time_vector = np.linspace(0, 60, processed_data.shape[1])
        else:
            print(f"Unexpected shape in {file_path}: {data.shape}")
            continue
        
        # 각 채널(33~40) 신호 플롯
        for ch in range(num_channels):
            ax.plot(time_vector, processed_data[ch, :], label=f"Ch {ch+33}", linewidth=1)
        ax.set_title(f"Trial {trial:02d}", fontsize=10)
        if trial % cols == 0:
            ax.set_ylabel("Signal", fontsize=8)
        if trial >= num_trials - cols:
            ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize='x-small', loc='upper right')
    
    # 남는 subplot 삭제
    for i in range(num_trials, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(f"Subject {subject} Trials Visualization", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"{subject}_trials_visualization.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization for {subject} at {save_path}")
