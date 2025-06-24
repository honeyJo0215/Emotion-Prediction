#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
save_regularized_ppg_all_subjects.py

DEAP PPG data (plethysmograph 채널)을 32명의 피험자(s01~s32)에 대해 자동으로 처리하여,
각 trial별로 정규화된 펄스 파형(8064,)을 NumPy 배열로 저장합니다.
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample

# === 설정 ===
INPUT_DIR    = "/home/bcml1/2025_EMOTION/DEAP_npy_files+label"
OUTPUT_BASE  = "/home/bcml1/2025_EMOTION/DEAP_PPG_regular_waveform"
PPG_CHANNEL  = 38        # plethysmograph 채널 인덱스 (0-based)
FPS          = 128.0     # 샘플링 레이트 (Hz)
BP_LOW       = 0.5       # Hz (저주파 베이스라인 제거)
BP_HIGH      = 5.0       # Hz (고주파 노이즈 제거)
# === 설정 끝 ===

def bandpass(sig, fs, lowcut, highcut, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, sig)

def regularize_ppg(sig):
    # 1) 밴드패스 필터
    filt = bandpass(sig, FPS, BP_LOW, BP_HIGH)
    # 2) 피크 검출 (최소 0.4초 간격, prominence = 표준편차)
    min_dist = int(FPS * 0.4)
    peaks, _ = find_peaks(filt, distance=min_dist, prominence=np.std(filt))
    if len(peaks) < 2:
        return None
    # 3) 중앙값 간격 길이
    intervals = np.diff(peaks)
    med_len = int(np.median(intervals))
    if med_len < 1:
        return None
    # 4) 펄스별 리샘플 -> 템플릿
    segments = []
    for j in range(len(peaks)-1):
        seg = filt[peaks[j]:peaks[j+1]]
        segments.append(resample(seg, med_len))
    template = np.mean(segments, axis=0)
    # 5) RMS 스케일링
    template *= np.std(filt) / (np.std(template) + 1e-8)
    # 6) 타일링하여 전체 길이만큼 반복
    n_tiles = int(np.ceil(len(sig) / med_len))
    regular = np.tile(template, n_tiles)[:len(sig)]
    return regular

def main():
    for subj_idx in range(1, 33):
        subid = f"s{subj_idx:02d}"
        input_path = os.path.join(INPUT_DIR, f"{subid}_signals.npy")
        if not os.path.isfile(input_path):
            print(f"[WARN] {input_path} not found, skip.")
            continue

        # 출력 디렉토리
        out_dir = os.path.join(OUTPUT_BASE, subid)
        os.makedirs(out_dir, exist_ok=True)

        # (40 trials × 40 channels × 8064 samples) 로드
        data = np.load(input_path)
        trials, channels, length = data.shape
        if PPG_CHANNEL < 0 or PPG_CHANNEL >= channels:
            print(f"[ERROR] Invalid PPG_CHANNEL for {subid}, skip.")
            continue

        # plethysmograph 채널만 추출
        ppg_all = data[:, PPG_CHANNEL, :]  # shape = (40, 8064)

        # 각 trial 처리
        for t in range(trials):
            raw = ppg_all[t].astype(float)
            reg = regularize_ppg(raw)
            if reg is None:
                print(f"[WARN] {subid} trial{t:02d}: 정규화 실패, skip.")
                continue

            out_fname = f"{subid}_trial{t:02d}_regular.npy"
            out_path  = os.path.join(out_dir, out_fname)
            np.save(out_path, reg)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
