import numpy as np
import os
import matplotlib.pyplot as plt
import glob

def visualize_eeg_signal(file_path, save_dir, width_scale=10):
    """
    EEG npy 파일을 시각화하여 지정된 경로에 저장하는 함수.
    
    파일명 형식: sXX_trial_XX_label_X.npy
      - 파일의 shape는 (60, 32, 128)입니다.
      - 60: segment 개수 → 각 채널별로 60개의 segment를 이어붙여 하나의 긴 파형으로 생성 (x축은 0~60초).
      - 32: 채널 수 → 각 채널에 해당하는 파형을 별도의 subplot으로 표시.
      - 최종적으로 32개의 subplot(채널 파형)이 아래로 순서대로 배치된 하나의 PNG 파일로 저장됨.
      
    저장 경로는 save_dir/DEAP_EEG3/sXX/ 폴더 내에, 원본 파일명을 기반으로 예를 들어,
    "s01_trial_02_label_1.png"와 같이 저장됩니다.
    
    width_scale를 통해 figure의 가로 크기를 확대합니다. (기본값 10: 원래보다 10배 넓게)
    """
    # npy 파일 로드
    data = np.load(file_path)  # data.shape: (60, 32, 128)
    
    # 파일명에서 피실험자, trial, label 정보 추출
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    subject_id = parts[0]              # 예: "s01"
    trial_id = parts[1]                # 예: "trial"
    trial_num = parts[2]               # 예: "02"
    label = parts[4].split('.')[0]     # 예: "1"
    
    # 출력 파일명 생성 (예: s01_trial_02_label_1.png)
    out_filename = f"{subject_id}_{trial_id}_{trial_num}_label_{label}.png"
    
    # 결과 저장할 디렉토리 생성: save_dir/DEAP_EEG3/sXX/
    subject_dir = os.path.join(save_dir, "DEAP_EEG3", subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    
    # data의 shape: (60, 32, 128)
    num_segments, num_channels, num_samples = data.shape
    
    # 각 채널별로 60개의 segment를 이어붙여 하나의 긴 파형(1차원 배열)을 생성
    channel_waveforms = []
    for ch in range(num_channels):
        # data[:, ch, :] shape: (60, 128) → concatenate along time axis → (60*128,)
        waveform = np.concatenate(data[:, ch, :], axis=-1)
        channel_waveforms.append(waveform)
    
    total_length = num_segments * num_samples  # 예: 7680
    # x축 좌표는 0~60초 범위로 생성 (시간 범위는 그대로 60초)
    x_axis = np.linspace(0, 60, total_length)
    
    # figure의 가로 크기를 width_scale 배 확장
    fig, axes = plt.subplots(num_channels, 1, figsize=(20 * width_scale, 2.5 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]
    
    for ch in range(num_channels):
        axes[ch].plot(x_axis, channel_waveforms[ch], color='b')
        axes[ch].set_ylabel(f'Ch {ch+1}', fontsize=10)
        axes[ch].grid(True, linestyle='--', alpha=0.5)
        # x축 눈금은 0~60초를 1초 단위로 표시
        axes[ch].set_xticks(np.arange(0, 61, 1))
    
    axes[-1].set_xlabel('Time (sec)', fontsize=12)
    fig.suptitle(f'EEG Waveforms for {subject_id} (Trial {trial_num}, Label {label})', fontsize=16)
    
    # 최종 이미지 저장 (PNG 파일)
    save_path = os.path.join(subject_dir, out_filename)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

# ============================
# 메인 처리: 모든 피실험자의 모든 EEG 파일 처리 (Subject 01~22)
# ============================

# 예시: npy 파일들이 저장된 입력 폴더 (예: "/home/bcml1/2025_EMOTION/DEAP_EEG3")
input_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG3"
# 파일 패턴: sXX_trial_XX_label_X.npy 형식의 파일들
file_pattern = os.path.join(input_dir, "s??_trial_*_label_*.npy")
file_list = glob.glob(file_pattern)

# 결과 저장 경로
save_dir = "/home/bcml1/sigenv/_4주차_data_check/DEAP_EEG_Visualize/"
os.makedirs(save_dir, exist_ok=True)

# Subject 번호가 01부터 22인 경우만 처리
for file_path in file_list:
    filename = os.path.basename(file_path)
    subject_str = filename.split('_')[0]  # 예: "s01"
    try:
        subject_num = int(subject_str[1:])  # "s01" -> 1
    except ValueError:
        continue  # 형식에 맞지 않으면 건너뛰기
    if 1 <= subject_num <= 22:
        output_path = visualize_eeg_signal(file_path, save_dir, width_scale=10)
        print("저장 완료:", output_path)
