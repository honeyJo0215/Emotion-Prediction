import os
import numpy as np
import re

# 세션별 라벨 (총 24개)
session1_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

# subject 파일들이 저장된 폴더 (총 15개)
subject_dir = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/3_npy'
# 샘플 파일들을 저장할 폴더
output_dir = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/3_npy_sample'
os.makedirs(output_dir, exist_ok=True)

# subject 폴더 내 npy 파일 목록을 정렬하여 가져옵니다.
subject_files = sorted([f for f in os.listdir(subject_dir) if f.endswith('.npy')])

for subject_file in subject_files:
    subject_path = os.path.join(subject_dir, subject_file)
    # npy 파일을 불러오며, 딕셔너리 형태임을 가정합니다.
    subject_data = np.load(subject_path, allow_pickle=True).item()

    # 정규식을 사용하여, 키가 어느 접두어가 있든 "eeg" 뒤에 숫자가 붙은 경우를 추출합니다.
    pattern = re.compile(r'.*eeg(\d+)$')
    session_keys = []
    for key in subject_data:
        match = pattern.match(key)
        if match:
            session_num = int(match.group(1))
            session_keys.append((session_num, key))
    # 세션 번호 기준으로 정렬 (예: 1부터 24까지)
    session_keys.sort(key=lambda x: x[0])

    # 각 세션(샘플)별로 분리 및 저장
    for session_num, key in session_keys:
        # 만약 session_num이 1~24 범위가 아니라면 건너뜁니다.
        if session_num < 1 or session_num > 24:
            print(f"Skipping key {key} with session number {session_num} out of expected range.")
            continue

        sample_data = subject_data[key]
        label = session1_label[session_num - 1]

        # 샘플 데이터와 라벨을 딕셔너리로 구성
        sample_dict = {'data': sample_data, 'label': label}

        # 파일 이름 구성: 예) subjectBase_sample_01_label_1.npy
        subject_base = os.path.splitext(subject_file)[0]
        sample_num_str = str(session_num).zfill(2)
        output_filename = f"{subject_base}_sample_{sample_num_str}_label_{label}.npy"
        output_path = os.path.join(output_dir, output_filename)

        np.save(output_path, sample_dict)
        print(f"Saved: {output_path}")
