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
output_dir = "~/sigenv/eeg_showdata3"
output_dir = os.path.expanduser(output_dir)

# 출력 디렉터리 생성 (없을 경우)
os.makedirs(output_dir, exist_ok=True)

# 데이터 파일 경로
file_positive = os.path.join(source_dir, "s01_positive_FB.npy")
file_negative = os.path.join(source_dir, "s01_negative_FB.npy")
label_file = os.path.join(label_dir, "s01_label.npy")

# 데이터 로드
data_positive = np.load(file_positive)
data_negative = np.load(file_negative)
labels = np.load(label_file) if os.path.exists(label_file) else None

# Bandpass filter 적용 함수
def apply_bandpass_filter(data, lowcut=0.5, highcut=30, sampling_rate=128):
    num_samples, num_channels, time_steps = data.shape
    filtered_data = np.zeros_like(data)
    for sample in range(num_samples):
        for channel in range(num_channels):
            filtered_data[sample, channel, :] = nk.signal_filter(
                data[sample, channel, :], lowcut=lowcut, highcut=highcut, sampling_rate=sampling_rate, method="butterworth"
            )
    return filtered_data

# Bandpass filter 적용
data_positive_filtered = apply_bandpass_filter(data_positive)
data_negative_filtered = apply_bandpass_filter(data_negative)

# 2. 채널별 시각화
def plot_channel_data(data, title, output_prefix, labels=None):
    num_samples, num_channels, time_steps = data.shape  # Shape 수정
    for channel in range(num_channels):
        plt.figure(figsize=(12, 6))
        for sample in range(num_samples):
            label_info = f" (Label: {labels[sample]})" if labels is not None else ""
            plt.plot(range(time_steps), data[sample, channel, :], alpha=0.6, label=f"Sample {sample}{label_info}")
        plt.title(f"{title} - Channel {channel}")
        plt.xlabel("Time Index")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right", fontsize="small", frameon=False)
        output_path = os.path.join(output_dir, f"{output_prefix}_channel_{channel}.png")
        plt.savefig(output_path)
        plt.close()

# 양성 및 부정 데이터 각각 시각화 (필터링된 데이터)
plot_channel_data(data_positive_filtered, "Positive EEG Data (Filtered)", "positive_filtered", labels=labels[:data_positive_filtered.shape[0]] if labels is not None else None)
plot_channel_data(data_negative_filtered, "Negative EEG Data (Filtered)", "negative_filtered", labels=labels[:data_negative_filtered.shape[0]] if labels is not None else None)

# 3. 샘플 간 평균 및 분산 비교
def compute_mean_variance(data):
    mean_per_sample = np.mean(data, axis=(1, 2))  # 샘플별 평균
    variance_per_sample = np.var(data, axis=(1, 2))  # 샘플별 분산
    return mean_per_sample, variance_per_sample

mean_positive, variance_positive = compute_mean_variance(data_positive_filtered)
mean_negative, variance_negative = compute_mean_variance(data_negative_filtered)

# 평균 및 분산 비교 시각화
plt.figure(figsize=(12, 6))
plt.plot(mean_positive, label="Positive Data Mean (Filtered)", marker='o')
plt.plot(mean_negative, label="Negative Data Mean (Filtered)", marker='x')
plt.title("Sample-wise Mean Comparison (Filtered)")
plt.xlabel("Sample Index")
plt.ylabel("Mean Amplitude")
plt.legend()
output_path = os.path.join(output_dir, "mean_comparison_filtered.png")
plt.savefig(output_path)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(variance_positive, label="Positive Data Variance (Filtered)", marker='o')
plt.plot(variance_negative, label="Negative Data Variance (Filtered)", marker='x')
plt.title("Sample-wise Variance Comparison (Filtered)")
plt.xlabel("Sample Index")
plt.ylabel("Variance")
plt.legend()
output_path = os.path.join(output_dir, "variance_comparison_filtered.png")
plt.savefig(output_path)
plt.close()
