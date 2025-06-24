import numpy as np
import glob
import os

# 📌 원본 EEG NPY 파일 폴더
eeg_npy_folder = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG/"
# labels_npy_folder = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_label/"
output_folder = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_segments/"  # 저장 폴더

# 📌 변환할 EEG NPY 파일 리스트 가져오기
eeg_files = glob.glob(os.path.join(eeg_npy_folder, "*.npy"))
print(f"🔍 총 {len(eeg_files)}개의 EEG NPY 파일을 찾았습니다.")

# 📌 저장 폴더 생성 (없을 경우 자동 생성)
os.makedirs(output_folder, exist_ok=True)

# 📌 각 파일을 변환하여 저장
for eeg_file in eeg_files:
    subject_id = os.path.basename(eeg_file).replace("_eeg.npy", "")  # 예: s01
    
    # 📌 EEG 데이터 로드
    eeg_data = np.load(eeg_file)  # Shape: (40, 8064, 32)
    
    # 📌 데이터 슬라이싱 (8064 -> 128씩 63개로 분할)
    num_segments = 63  # 63개의 1초 segment 생성
    segmented_data = np.split(eeg_data, num_segments, axis=1)  # Shape: (40, 128, 32) * 63
    
    # 📌 각 segment를 개별 NPY 파일로 저장
    for i, segment in enumerate(segmented_data):
        segment_file_path = os.path.join(output_folder, f"{subject_id}_segment_{i:03d}.npy")
        np.save(segment_file_path, segment)
        print(f"✅ {segment_file_path} 저장 완료")

print("🎉 모든 변환이 완료되었습니다!")