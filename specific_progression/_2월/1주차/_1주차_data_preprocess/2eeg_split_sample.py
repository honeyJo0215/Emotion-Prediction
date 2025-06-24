import numpy as np
import glob
import os

# 📌 원본 1초 단위 분할된 EEG NPY 파일 폴더
segmented_eeg_folder = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_segments/"
output_folder = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_segments_samples/"  # 저장 폴더

# 📌 변환할 NPY 파일 리스트 가져오기
segmented_files = glob.glob(os.path.join(segmented_eeg_folder, "*.npy"))
print(f"🔍 총 {len(segmented_files)}개의 1초 단위 NPY 파일을 찾았습니다.")

# 📌 저장 폴더 생성 (없을 경우 자동 생성)
os.makedirs(output_folder, exist_ok=True)

# 📌 각 파일을 변환하여 저장
for seg_file in segmented_files:
    filename = os.path.basename(seg_file)  # 파일명 가져오기
    subject_id, segment_id = filename.split("_segment_")  # 주제 ID(sXX) 및 세그먼트 번호 추출
    segment_id = segment_id.replace(".npy", "")  # 확장자 제거
    
    # 📌 EEG 데이터 로드 (Shape: (40, 128, 32))
    eeg_data = np.load(seg_file)
    
    # 📌 각 샘플별로 개별 저장 (sample이 먼저 오도록 파일명 변경)
    for sample_idx in range(eeg_data.shape[0]):  # 40개 샘플 반복
        sample_data = eeg_data[sample_idx]  # Shape: (128, 32)
        sample_file_path = os.path.join(output_folder, f"{subject_id}_sample_{sample_idx:02d}_segment_{segment_id}.npy")
        np.save(sample_file_path, sample_data)
        print(f"✅ {sample_file_path} 저장 완료")

print("🎉 모든 변환이 완료되었습니다!")
