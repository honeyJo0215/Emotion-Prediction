import numpy as np
import os
import matplotlib.pyplot as plt
import glob

def visualize_ppg_signal(file_path, save_dir):
    """
    ppg npy 파일을 시각화하여 지정된 경로에 저장하는 함수.
    
    파일명 형식: sXX_ppg_signals_segment_XX.npy
      - (40, 128) 배열인 경우: 40 trial의 파형(채널 1개)을 각각 subplot으로 그려 하나의 jpg로 저장.
      - 만약 파일이 3차원(40, n채널, 128)인 경우: 각 채널의 모든 trial 파형을 한 subplot에 오버레이하여,
        n개의 subplot이 있는 하나의 jpg 파일로 저장.
        
    저장 경로는 save_dir/sXX/ 폴더 내에 ppg_segment_XX.jpg 파일로 저장됨.
    """
    # npy 파일 로드
    data = np.load(file_path)
    
    # 파일명에서 피실험자(sXX)와 segment 번호 추출
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    subject_id = parts[0]  # 예: "s01"
    segment_id = parts[-1].split('.')[0]  # 예: "07"
    
    # 결과 저장할 피실험자 디렉토리 생성
    subject_dir = os.path.join(save_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    
    # 데이터 차원에 따라 처리
    if data.ndim == 2:
        # data shape: (num_trials, num_samples) -> 채널 1개
        num_trials, num_samples = data.shape
        
        # 각 trial의 파형을 subplot으로 그리기 (세로로 배열)
        fig, axes = plt.subplots(num_trials, 1, figsize=(10, 2*num_trials), sharex=True)
        if num_trials == 1:
            axes = [axes]
            
        for i in range(num_trials):
            axes[i].plot(data[i], color='b')
            axes[i].set_ylabel(f'Trial {i+1}')
            axes[i].grid(True, linestyle='--', alpha=0.5)
        
        axes[-1].set_xlabel('Time (sample index)')
        fig.suptitle(f'PPG Signals for {subject_id}, Segment {segment_id}', fontsize=16)
        
    elif data.ndim == 3:
        # data shape: (num_trials, num_channels, num_samples)
        num_trials, num_channels, num_samples = data.shape
        
        # 각 채널별 subplot 생성 (채널 수만큼)
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 3*num_channels), sharex=True)
        if num_channels == 1:
            axes = [axes]
        
        for ch in range(num_channels):
            for trial in range(num_trials):
                # 각 trial의 파형을 투명도(alpha)를 주어 오버레이
                axes[ch].plot(data[trial, ch, :], color='b', alpha=0.7)
            axes[ch].set_title(f'Channel {ch+1}')
            axes[ch].grid(True, linestyle='--', alpha=0.5)
        axes[-1].set_xlabel('Time (sample index)')
        fig.suptitle(f'PPG Signals for {subject_id}, Segment {segment_id}', fontsize=16)
        
    else:
        raise ValueError("데이터 차원이 예상과 다릅니다.")
    
    # 결과 파일 저장 (jpg)
    save_path = os.path.join(subject_dir, f'ppg_segment_{segment_id}.jpg')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

# ============================
# 메인 처리: 모든 피실험자의 모든 세그먼트 파일 처리
# ============================

# npy 파일들이 저장된 입력 폴더 (예: "/home/bcml1/2025_EMOTION/DEAP_PPG")
input_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG"
# 파일 패턴: sXX_ppg_signals_segment_XX.npy 형식의 파일들
file_pattern = os.path.join(input_dir, "s??_ppg_signals_segment_*.npy")
file_list = glob.glob(file_pattern)

# 결과 저장 경로
save_dir = "/home/bcml1/sigenv/_4주차_data_check/DEAP_PPG_Visualize/"
os.makedirs(save_dir, exist_ok=True)

# 모든 파일에 대해 시각화 수행
for file_path in file_list:
    output_path = visualize_ppg_signal(file_path, save_dir)
    print("저장 완료:", output_path)
