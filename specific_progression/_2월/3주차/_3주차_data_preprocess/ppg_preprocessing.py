import os
import numpy as np

# 입력 및 출력 경로 설정
input_dir = "/home/bcml1/2025_EMOTION/DEAP_npy_files+label/"
output_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_10s/"

os.makedirs(output_dir, exist_ok=True)

# 설정 값
discard_samples = 384      # 앞 3초: 4*128
desired_length = 7680       # 60초: 60*128
segment_length = 1280       # 10초: 10*128
num_segments = 6

# 처리할 피험자 수 (s01 ~ s32)
for subject_id in range(1, 33):
    subject_filename = f"s{subject_id:02d}_signals.npy"
    subject_filepath = os.path.join(input_dir, subject_filename)

    if not os.path.exists(subject_filepath):
        print(f"파일을 찾을 수 없음: {subject_filepath}")
        continue

    # 데이터 로드 및 PPG 채널(39번째) 추출
    subject_data = np.load(subject_filepath, allow_pickle=True)
    ppg_data = subject_data[:, 38, :]

    # 앞 4초(512 샘플) 버림 후 60초(7680 샘플) 구간 선택
    start_index = discard_samples
    end_index = start_index + desired_length
    # 만약 데이터가 부족하면 0으로 패딩
    if ppg_data.shape[1] < end_index:
        ppg_slice = ppg_data[:, start_index:]
        pad_width = end_index - ppg_slice.shape[1]
        ppg_slice = np.pad(ppg_slice, ((0, 0), (0, pad_width)), mode='constant')
    else:
        ppg_slice = ppg_data[:, start_index:end_index]

    # 60초 데이터(7680 샘플)를 10초(1280 샘플) 단위로 6개 세그먼트로 분할
    ppg_segments = [ppg_slice[:, i * segment_length:(i + 1) * segment_length]
                    for i in range(num_segments)]

    # 분할된 세그먼트 저장
    for seg_idx, segment in enumerate(ppg_segments):
        segment_filename = f"s{subject_id:02d}_ppg_signals_segment_{seg_idx:02d}.npy"
        segment_filepath = os.path.join(output_dir, segment_filename)
        np.save(segment_filepath, segment)

    print(f"✅ {subject_filename} 변환 완료: {len(ppg_segments)}개 세그먼트 저장됨.")

print("🎯 모든 데이터 변환 완료! 변환된 파일이 DEAP_PPG_10s 폴더에 저장되었습니다.")
