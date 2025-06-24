import os
import glob
import numpy as np
from mne.decoding import CSP

# 1. 한 폴더 내의 npy 파일들을 로드하면서 라벨도 추출하는 함수
def load_eeg_files(folder):
    files = glob.glob(os.path.join(folder, '*.npy'))
    data_list = []
    labels = []
    for f in sorted(files):
        # 예: "1_20160518_sample_01_label_1.npy"
        base = os.path.basename(f)
        # '_'로 split 후 마지막 부분(확장자 제거)에서 라벨 추출
        label = int(base.split('_')[-1].split('.')[0])
        labels.append(label)
        data = np.load(f)  # shape: (62, T)
        data_list.append(data)
    return data_list, labels, sorted(files)

# 2. 한 npy 파일의 데이터를 1초(200포인트) 단위로 분할하는 함수  
#    단, 첫번째 포인트(잡신호)는 제외하여 data[:, 1:] 사용
def segment_data(eeg_data, fs=200):
    # eeg_data shape: (62, T)
    effective_data = eeg_data[:, 1:]  # 첫 번째 잡신호 제거
    n_points = effective_data.shape[1]
    n_seconds = n_points // fs  # 정수 초 단위
    segments = []
    for i in range(n_seconds):
        seg = effective_data[:, i*fs:(i+1)*fs]  # 각 segment: (62, 200)
        segments.append(seg)
    return segments

# 3. subject 폴더 내 모든 파일에서 segment와 해당 라벨을 모으기
def gather_segments_and_labels(data_list, labels, fs=200):
    all_segments = []  # 각 segment: (62,200)
    seg_labels = []    # segment에 해당하는 라벨 (파일 단위라 동일 label 반복)
    for data, lab in zip(data_list, labels):
        segs = segment_data(data, fs)
        all_segments.extend(segs)
        seg_labels.extend([lab] * len(segs))
    return np.array(all_segments), np.array(seg_labels)  # X shape: (n_trials, 62, 200)

# 4. mne의 CSP 클래스를 이용하여 subject 내 CSP 필터 학습
def compute_csp(X, y, n_components=4):
    # CSP 객체 생성 (n_components: 사용할 필터 수, log=True이면 로그-분산을 계산)
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    csp.fit(X, y)
    return csp

# 5. 학습된 CSP 필터를 각 파일의 segment에 적용해 feature 추출
def extract_csp_features(csp, segments):
    features = []
    for seg in segments:
        # mne의 transform은 (trials, channels, samples) 입력을 받으므로 np.newaxis 추가
        feat = csp.transform(seg[np.newaxis, :, :])
        # feat shape: (1, n_components) → squeeze하여 (n_components,)
        features.append(feat.squeeze())
    return features  # list of feature vectors per second

# 6. subject 단위로 전체 프로세스를 수행하는 함수  
#    - 해당 subject 폴더 내 모든 파일을 이용해 CSP 필터를 학습한 후, 각 파일별로 초단위 CSP feature를 저장
def process_subject(raw_folder, save_folder, fs=200, n_components=4):
    # raw data load
    data_list, file_labels, file_paths = load_eeg_files(raw_folder)
    
    # 전체 파일의 segment와 label을 모아서 CSP 필터 학습에 사용
    X, y = gather_segments_and_labels(data_list, file_labels, fs)
    csp = compute_csp(X, y, n_components=n_components)
    
    # 각 파일별로 처리: 원본 파일명과 동일하게 CSP feature들을 npy로 저장
    for data, f in zip(data_list, file_paths):
        segments = segment_data(data, fs)
        features = extract_csp_features(csp, segments)
        # features는 각 초(segment)마다 n_components 길이의 feature vector의 리스트
        save_path = os.path.join(save_folder, os.path.basename(f))
        np.save(save_path, features)
        print(f"Saved CSP features: {save_path}")

# 7. 메인 함수: subject 1, 2, 3 폴더에 대해 처리
def main():
    subjects = ['1', '2', '3']
    base_raw = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data'
    base_csp = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_CSP'
    
    for subj in subjects:
        raw_folder = os.path.join(base_raw, f"{subj}_npy_sample")
        save_folder = os.path.join(base_csp, subj)
        os.makedirs(save_folder, exist_ok=True)
        print(f"Processing subject {subj} ...")
        process_subject(raw_folder, save_folder, fs=200, n_components=4)
    print("CSP feature extraction 완료.")

if __name__ == '__main__':
    main()
