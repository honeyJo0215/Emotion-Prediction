import numpy as np
import glob
import os
from scipy.signal import butter, filtfilt

# 📌 Band 정의
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

# 📌 샘플링 레이트 설정 (DEAP 데이터 기준)
SAMPLING_RATE = 128  # Hz

# 📌 Band-Pass 필터 함수
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)

# 📌 원본 EEG NPY 파일 폴더
input_folder = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_segments_samples/"
output_folder = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_sam_seg_bands/"  # 저장 폴더

# 📌 변환할 NPY 파일 리스트 가져오기
eeg_files = glob.glob(os.path.join(input_folder, "*.npy"))
print(f"🔍 총 {len(eeg_files)}개의 NPY 파일을 찾았습니다.")

# 📌 저장 폴더 생성 (없을 경우 자동 생성)
os.makedirs(output_folder, exist_ok=True)

# 📌 각 파일을 변환하여 저장
for eeg_file in eeg_files:
    filename = os.path.basename(eeg_file)  # 파일명 가져오기
    subject_id = filename.replace(".npy", "")  # 확장자 제거

    # 📌 EEG 데이터 로드 (Shape: (128, 32))
    eeg_data = np.load(eeg_file)  

    # 📌 Band-Pass 필터 적용 (결과 Shape: (4, 128, 32))
    band_data = np.zeros((4, 128, 32))  # 4 Bands, 128 Timesteps, 32 Channels

    for i, (band, (low, high)) in enumerate(BANDS.items()):
        band_data[i] = bandpass_filter(eeg_data, low, high, SAMPLING_RATE)

    # 📌 필터링된 데이터 저장
    output_file_path = os.path.join(output_folder, f"{subject_id}_bands.npy")
    np.save(output_file_path, band_data)
    print(f"✅ {output_file_path} 저장 완료")

print("🎉 모든 변환이 완료되었습니다!")
