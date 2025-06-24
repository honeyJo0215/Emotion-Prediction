#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_ppg_signals.py

이 스크립트는 40개의 10초 길이 PPG 시그널(샘플당 1280 포인트)이 들어있는
NumPy 배열을 로드한 뒤, 각 샘플을 개별 PNG로 시각화하여 지정된 디렉토리에 저장합니다.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# === 설정 (필요에 따라 수정) ===
INPUT_NPY     = "/home/bcml1/2025_EMOTION/DEAP_PPG_regular_waveform/s01/s01_trial00_regular.npy"
SAMPLING_RATE = 128.0  # 10초 동안 1280 샘플 → 128 Hz
OUTPUT_DIR    = "/home/bcml1/sigenv/_4월/_rppg/ppg_visualize"
# === 설정 끝 ===

def main():
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # PPG 데이터 로드 (shape = (40, 1280))
    ppg_data = np.load(INPUT_NPY)
    num_signals = 1 
    length, = ppg_data.shape

    # 시간 축 생성 (초 단위)
    time_axis = np.arange(length) / SAMPLING_RATE

    # 파일 기본 이름
    base_name = os.path.splitext(os.path.basename(INPUT_NPY))[0]

    # 각 샘플별로 플롯 및 저장
    for idx in range(num_signals):
        signal = ppg_data

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, signal, linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("PPG Amplitude (a.u.)")
        plt.title(f"{base_name} - Sample {idx:02d}")
        plt.grid(True)
        plt.tight_layout()

        out_fname = f"{base_name}_sample{idx:02d}.png"
        out_path  = os.path.join(OUTPUT_DIR, out_fname)
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved PPG plot: {out_path}")

if __name__ == "__main__":
    main()
