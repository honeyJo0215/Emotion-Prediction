import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

# 데이터 디렉토리 설정
source_dir = "~/2025_EMOTION/DEAP_EEG/ch_BPF"
source_dir = os.path.expanduser(source_dir)

label_dir = "~/2025_EMOTION/DEAP_EEG/labels"
label_dir = os.path.expanduser(label_dir)

# 결과 저장 디렉터리 설정
output_dir = "~/sigenv/eeg_showdata4"
output_dir = os.path.expanduser(output_dir)

# 출력 디렉터리 생성 (없을 경우)
os.makedirs(output_dir, exist_ok=True)

# Bandpass filter 적용 함수
def apply_bandpass_filter(data, lowcut=4, highcut=45, sampling_rate=256):   #샘플링 속도 128->256으로 변경
    num_samples, num_channels, time_steps = data.shape
    filtered_data = np.zeros_like(data)
    for sample in range(num_samples):
        for channel in range(num_channels):
            filtered_data[sample, channel, :] = nk.signal_filter(
                data[sample, channel, :], lowcut=lowcut, highcut=highcut, sampling_rate=sampling_rate, method="butterworth"
            )
    return filtered_data

# 파일 처리 함수
def process_eeg_file(eeg_file, label_file, subject_id):
    print(f"Processing: {eeg_file}")

    # EEG 데이터 로드
    eeg_data = np.load(eeg_file)  # (num_samples, num_channels, time_steps) 형식 가정
    print(f"EEG Data Shape: {eeg_data.shape}")

    # 상태(positive/negative) 확인
    condition = "positive" if "positive" in eeg_file else "negative"

    # 라벨 로드 또는 기본 라벨 적용
    if os.path.exists(label_file):
        labels = np.load(label_file)
        if len(labels) != eeg_data.shape[0]:
            print(f"Warning: Label count ({len(labels)}) does not match sample count ({eeg_data.shape[0]}). Using default label 'label0'.")
            labels = ["label0"] * eeg_data.shape[0]
    else:
        print(f"Label file not found for {subject_id}. Using default label 'label0'.")
        labels = ["label0"] * eeg_data.shape[0]

    # Bandpass filter 적용
    filtered_data = apply_bandpass_filter(eeg_data)

    # 채널별 시각화
    for sample_idx in range(filtered_data.shape[0]):
        for channel in range(filtered_data.shape[1]):
            plt.figure(figsize=(12, 6))
            plt.plot(range(filtered_data.shape[2]), filtered_data[sample_idx, channel, :], alpha=0.8, label=f"Channel {channel+1}")
            plt.title(f"EEG Signal (Filtered) - Subject {subject_id} - Sample {sample_idx+1} - Channel {channel+1} (Label: {labels[sample_idx]})")
            plt.xlabel("Time Index")
            plt.ylabel("Amplitude")
            plt.legend()

            # 그래프 저장
            output_path = os.path.join(output_dir, f"{subject_id}_{condition}_sample_{sample_idx+1}_channel_{channel+1}_label_{labels[sample_idx]}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")

# 모든 파일 처리
for file_name in sorted(os.listdir(source_dir)):
    if file_name.endswith(".npy"):
        eeg_file = os.path.join(source_dir, file_name)
        subject_id = file_name.split("_")[0]  # 예: "s01_positive_FB.npy" -> "s01"
        label_file = os.path.join(label_dir, f"{subject_id}_label.npy")

        try:
            process_eeg_file(eeg_file, label_file, subject_id)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
