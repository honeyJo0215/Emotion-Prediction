import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 데이터 디렉토리 설정
data_dir = "~/2025_EMOTION/DEAP_EEG/ch_BPF"
data_dir = os.path.expanduser(data_dir)

# 결과 저장 디렉토리 설정
output_dir = "~/sigenv/eeg_showdata1"
output_dir = os.path.expanduser(output_dir)
os.makedirs(output_dir, exist_ok=True)

# 디렉토리 내 파일 목록 가져오기
files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

# 시각화를 위한 데이터 로드 함수
def load_and_plot_3d(file_path, title):
    data = np.load(file_path)  # 데이터 로드
    if data.ndim != 3:
        print(f"Skipping {title}: Data is not 3D.")
        return

    y = np.arange(data.shape[0])
    x = np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(data.shape[2]):
        Z = data[:, :, i]
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Amplitude")

    # 결과를 파일로 저장
    output_file = os.path.join(output_dir, f"{title}_3d_plot.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved 3D plot to {output_file}")

# 모든 파일 시각화
def visualize_all_files():
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        print(f"Visualizing: {file_name}")
        load_and_plot_3d(file_path, title=file_name)

# 실행
def main():
    print(f"Found {len(files)} .npy files in {data_dir}")
    visualize_all_files()

if __name__ == "__main__":
    main()
