#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_rppg_fixed.py

이 스크립트는 고정된 rPPG .npy 파일을 읽어 시간-시그널 플롯을
생성한 뒤, 지정된 경로에 PNG로 저장합니다.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# === 설정 (여기만 수정) ===
INPUT_PATH = "/home/bcml1/sigenv/_4월/_rppg/minipatch_rppg/s01/s01_trial01_rppg.npy"
FPS = 25.0
OUTPUT_DIR = "/home/bcml1/sigenv/_4월/_rppg/visualize_output"
# === 설정 끝 ===

def main():
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # rPPG 신호 로드
    rppg_signal = np.load(INPUT_PATH)
    num_samples = rppg_signal.shape[0]

    # 시간 축 생성 (초 단위)
    time_axis = np.arange(num_samples) / FPS

    # 출력 파일명 결정
    base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
    output_path = os.path.join(OUTPUT_DIR, base_name + ".png")

    # 플롯 생성
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, rppg_signal, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("rPPG Amplitude (a.u.)")
    plt.title(f"rPPG Signal: {base_name}")
    plt.grid(True)
    plt.tight_layout()

    # PNG 저장
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"rPPG visualization saved to: {output_path}")

if __name__ == "__main__":
    main()
