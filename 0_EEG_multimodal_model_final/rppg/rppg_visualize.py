import os
import numpy as np
import matplotlib.pyplot as plt

def plot_rppg(npy_path, save_path=None):
    """
    npy_path: 절대 경로로 지정한 .npy 파일 경로 (shape: [n_components, n_frames])
    save_path: (선택) 저장할 이미지 파일 경로. None이면 plt.show()로 화면에 표시만 함.
    """
    # 데이터 로드
    data = np.load(npy_path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    n_components, n_frames = data.shape
    time = np.arange(n_frames)

    # figure + subplots 생성
    fig, axes = plt.subplots(
        n_components, 1,
        figsize=(12, 2.5 * n_components),
        sharex=True
    )
    if n_components == 1:
        axes = [axes]

    # 각 component plot
    for idx, ax in enumerate(axes):
        ax.plot(time, data[idx])
        ax.set_ylabel(f"Comp {idx+1}")
        ax.grid(True, linestyle=':', linewidth=0.5)
    axes[-1].set_xlabel("Frame index")

    # 제목에 파일명 표시
    fig.suptitle(os.path.basename(npy_path), y=1.02)
    plt.tight_layout()

    if save_path:
        # 디렉토리 없으면 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved RPPG plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    # 1) 시각화할 .npy 파일 절대 경로
    npy_path = "/home/bcml1/sigenv/_7월/_rppg/minipatch_rppg_csp/s01/trial01_csp_rppg_multiclass.npy"
    # 2) 저장할 이미지 파일 경로 (예: .png, .jpg)
    save_path = "/home/bcml1/sigenv/_7월/rppg/plots/s01_trial01_rppg.png"
    # 저장하지 않고 화면에만 띄우고 싶으면 save_path=None 으로 호출하세요.
    plot_rppg(npy_path, save_path)
