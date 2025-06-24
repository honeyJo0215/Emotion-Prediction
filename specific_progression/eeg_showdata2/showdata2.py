import os
import numpy as np
import matplotlib.pyplot as plt

# 데이터 디렉토리 설정
source_dir = "~/2025_EMOTION/DEAP_EEG/ch_BPF"
source_dir = os.path.expanduser(source_dir)

# 결과 저장 디렉터리 설정
output_dir = "~/sigenv/eeg_showdata2"
output_dir = os.path.expanduser(output_dir)

# 출력 디렉터리 생성 (없을 경우)
os.makedirs(output_dir, exist_ok=True)

# 데이터 파일 경로
file_positive = os.path.join(source_dir, "s01_positive_FB.npy")
file_negative = os.path.join(source_dir, "s01_negative_FB.npy")

# 데이터 로드
data_positive = np.load(file_positive)
data_negative = np.load(file_negative)

# 2. 채널별 시각화
def plot_channel_data(data, title, output_prefix):
    num_samples, num_channels, time_steps = data.shape  # Shape 수정
    for channel in range(num_channels):
        plt.figure(figsize=(12, 6))
        for sample in range(num_samples):
            plt.plot(range(time_steps), data[sample, channel, :], alpha=0.6)  # 인덱싱 수정
        plt.title(f"{title} - Channel {channel}")
        plt.xlabel("Time Index")
        plt.ylabel("Amplitude")
        output_path = os.path.join(output_dir, f"{output_prefix}_channel_{channel}.png")
        plt.savefig(output_path)
        plt.close()

# 양성 및 부정 데이터 각각 시각화
plot_channel_data(data_positive, "Positive EEG Data", "positive")
plot_channel_data(data_negative, "Negative EEG Data", "negative")

# 3. 샘플 간 평균 및 분산 비교
def compute_mean_variance(data):
    mean_per_sample = np.mean(data, axis=(1, 2))  # 샘플별 평균
    variance_per_sample = np.var(data, axis=(1, 2))  # 샘플별 분산
    return mean_per_sample, variance_per_sample

mean_positive, variance_positive = compute_mean_variance(data_positive)
mean_negative, variance_negative = compute_mean_variance(data_negative)

# 평균 및 분산 비교 시각화
plt.figure(figsize=(12, 6))
plt.plot(mean_positive, label="Positive Data Mean", marker='o')
plt.plot(mean_negative, label="Negative Data Mean", marker='x')
plt.title("Sample-wise Mean Comparison")
plt.xlabel("Sample Index")
plt.ylabel("Mean Amplitude")
plt.legend()
output_path = os.path.join(output_dir, "mean_comparison.png")
plt.savefig(output_path)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(variance_positive, label="Positive Data Variance", marker='o')
plt.plot(variance_negative, label="Negative Data Variance", marker='x')
plt.title("Sample-wise Variance Comparison")
plt.xlabel("Sample Index")
plt.ylabel("Variance")
plt.legend()
output_path = os.path.join(output_dir, "variance_comparison.png")
plt.savefig(output_path)
plt.close()
