#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_regular_pulse_ppg.py

DEAP 데이터에서 plethysmograph 채널의 원신호를 베이스라인 이동 및 노이즈를 제거한 뒤
정규 펄스 템플릿을 생성하고, 이를 타일링한 정규 펄스 파형과 함께 PNG로 저장합니다.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, resample

# === 설정 ===
INPUT_NPY   = "/home/bcml1/2025_EMOTION/DEAP_npy_files+label/s01_signals.npy"
PPG_CHANNEL = 38             # plethysmograph 채널 인덱스 (0-based)
FPS         = 128.0          # 샘플링 레이트 (Hz)
OUTPUT_DIR  = "/home/bcml1/sigenv/_4월/_rppg/ppg2_visualize_output"
# 베이스라인 제거용 하이패스/밴드패스
HP_CUTOFF   = 0.5            # Hz (baseline wander 제거)
BP_LOW      = 0.5            # Hz (최소 PPG 주파수)
BP_HIGH     = 5.0            # Hz (최대 PPG 주파수)
# === 설정 끝 ===

def bandpass(sig, fs, lowcut, highcut, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, sig)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = np.load(INPUT_NPY)               # shape=(40,40,8064)
    trials, channels, length = data.shape

    if not (0 <= PPG_CHANNEL < channels):
        raise ValueError(f"Invalid PPG_CHANNEL: {PPG_CHANNEL}")

    ppg = data[:, PPG_CHANNEL, :]           # shape=(40,8064)
    time = np.arange(length) / FPS
    base = os.path.splitext(os.path.basename(INPUT_NPY))[0]

    for i in range(trials):
        sig = ppg[i].astype(float)

        # 1) 밴드패스 필터로 베이스라인 및 고주파 노이즈 제거
        filtered = bandpass(sig, FPS, BP_LOW, BP_HIGH)

        # 2) 피크 검출 (심박 최소 40 BPM -> 간격 최대 3초)
        min_dist = int(FPS * 0.4)  # 최소 0.4초 간격
        peaks, _ = find_peaks(filtered, distance=min_dist, prominence=np.std(filtered))
        if len(peaks) < 3:
            print(f"Trial {i:02d}: peaks too few, skipping")
            continue

        # 3) 펄스 간격 중앙값
        intervals = np.diff(peaks)
        med_len = int(np.median(intervals))

        # 4) 펄스별 세그먼트 추출 & 정규화된 템플릿 생성
        segments = []
        for j in range(len(peaks)-1):
            seg = filtered[peaks[j]:peaks[j+1]]
            # 리샘플해서 길이 통일
            segments.append(resample(seg, med_len))
        template = np.mean(segments, axis=0)
        # 템플릿 크기를 원 신호 RMS에 맞춰 스케일링
        template *= np.std(filtered) / (np.std(template) + 1e-8)

        # 5) 템플릿 타일링 -> 정규 펄스
        n_tiles = int(np.ceil(length / med_len))
        regular = np.tile(template, n_tiles)[:length]

        # 6) 시각화 및 저장
        plt.figure(figsize=(10,4))
        plt.plot(time, filtered, label="Filtered PPG", color='C0', linewidth=1)
        plt.plot(time, regular, label="Regular Pulse", color='C1', linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"{base} Trial {i:02d} - Regularized Pulse")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()

        out_name = f"{base}_trial{i:02d}_regular.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
