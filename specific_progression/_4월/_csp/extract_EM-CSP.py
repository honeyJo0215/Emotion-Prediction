import os
import re
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Butterworth bandpass filter 적용.
    data: (channels, time)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def compute_average_covariance(trials):
    """
    여러 trial (각 trial: (channels, T))에 대해 정규화된 평균 공분산 행렬 계산.
    각 trial에 대해 cov = X*X.T / trace(X*X.T)를 계산 후 평균.
    """
    cov_sum = None
    for X in trials:
        cov = np.dot(X, X.T)
        cov = cov / np.trace(cov)
        if cov_sum is None:
            cov_sum = cov
        else:
            cov_sum += cov
    avg_cov = cov_sum / len(trials)
    return avg_cov

def compute_csp_filters(covA, covB, n_components=8):
    """
    두 클래스(클래스 A와 B)의 평균 공분산 행렬을 이용해 CSP 필터 계산.
    표준 CSP 방법 (whitening transform 및 eigen-decomposition)을 사용하여,
    상위 n_components/2와 하위 n_components/2 (총 n_components)의 필터를 추출함.
    반환 W_selected: (n_components, channels)
    """
    R = covA + covB
    # R의 eigen-decomposition (정방행렬이므로 np.linalg.eigh 사용)
    eigvals, eigvecs = np.linalg.eigh(R)
    # 오름차순 정렬
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Whitening transform: P = Λ^(-1/2) * U^T
    P = np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T)
    # Whitened covariance for class A:
    S = np.dot(P, np.dot(covA, P.T))
    # S의 eigen-decomposition
    eigvals_S, eigvecs_S = np.linalg.eigh(S)
    # 내림차순 정렬
    sort_idx = np.argsort(eigvals_S)[::-1]
    eigvals_S = eigvals_S[sort_idx]
    eigvecs_S = eigvecs_S[:, sort_idx]
    # Projection matrix: W = (eigvecs_S.T) * P
    W = np.dot(eigvecs_S.T, P)
    # 상위 n_components/2와 하위 n_components/2 필터 선택 (예: 4 + 4 = 8)
    n_half = n_components // 2
    W_selected = np.vstack((W[:n_half, :], W[-n_half:, :]))
    return W_selected  # shape: (n_components, channels)

def parse_file_info(filename):
    """
    파일명 예: "1_20160518_sample_01_label_1.npy"
    subject, sample, label 값을 추출.
    """
    subject_match = re.match(r'(\d+)_', filename)
    subject = subject_match.group(1) if subject_match else 'unknown'
    
    sample_match = re.search(r'sample_(\d+)', filename)
    sample = sample_match.group(1) if sample_match else 'unknown'
    
    label_match = re.search(r'label_(\d+)', filename)
    label = label_match.group(1) if label_match else 'unknown'
    
    return subject, sample, label

def main():
    # 경로 설정
    raw_data_root = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data'
    output_root = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP'
    os.makedirs(output_root, exist_ok=True)
    fs = 200
    n_components = 8
    # 주파수 대역 설정 (Hz)
    bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 50)
    }
    
    # --- 1단계: 그룹화 ---
    # 각 실험 폴더(1_npy_sample ~ 3_npy_sample) 내에서 subject별, label별로
    # 주파수 대역별 필터링된 trial 데이터를 저장 (목표: 그룹 내 평균 공분산 행렬 계산)
    group_data = {}  # 구조: group_data[folder_num][subject][label][band] = [trial_data, ...]
    
    for folder_num in range(1, 4):
        folder_name = f'{folder_num}_npy_sample'
        folder_path = os.path.join(raw_data_root, folder_name)
        if not os.path.isdir(folder_path):
            print(f"폴더가 존재하지 않습니다: {folder_path}")
            continue
        group_data[folder_num] = {}
        for filename in os.listdir(folder_path):
            if not filename.endswith('.npy'):
                continue
            subject, sample, label = parse_file_info(filename)
            if subject not in group_data[folder_num]:
                group_data[folder_num][subject] = {}
            if label not in group_data[folder_num][subject]:
                # 각 label에 대해 각 주파수 대역에 빈 리스트 초기화
                group_data[folder_num][subject][label] = {band: [] for band in bands}
            file_path = os.path.join(folder_path, filename)
            try:
                data = np.load(file_path)  # shape: (channels, T)
            except Exception as e:
                print(f"파일 로드 오류 {file_path}: {e}")
                continue
            for band_name, (low, high) in bands.items():
                filtered = bandpass_filter(data, low, high, fs)
                group_data[folder_num][subject][label][band_name].append(filtered)
    
    # --- 2단계: CSP 필터 계산 (One-vs-All) ---
    # subject별로, 각 label에 대해 해당 label(trials_A)와 나머지(label != target, trials_B)를 사용해
    # 각 주파수 대역별 CSP 필터를 계산하여 저장.
    filters_dict = {}  # 구조: filters_dict[folder_num][subject][label][band] = W (shape: (8, channels))
    
    for folder_num in group_data:
        filters_dict[folder_num] = {}
        for subject in group_data[folder_num]:
            filters_dict[folder_num][subject] = {}
            subject_labels = group_data[folder_num][subject].keys()
            for label in subject_labels:
                filters_dict[folder_num][subject][label] = {}
                for band_name in bands:
                    # trials for 현재 label (class A)
                    trials_A = group_data[folder_num][subject][label][band_name]
                    # trials for 나머지 label들 (class B)
                    trials_B = []
                    for other_label in subject_labels:
                        if other_label == label:
                            continue
                        trials_B.extend(group_data[folder_num][subject][other_label][band_name])
                    if len(trials_A) == 0 or len(trials_B) == 0:
                        print(f"대상 부족: folder {folder_num}, subject {subject}, label {label}, band {band_name}")
                        # 기본값: 항등 행렬의 일부 (8 x channels)
                        # 여기서 data.shape[0]는 채널 수 (예, 62)로 가정
                        dummy_channels = trials_A[0].shape[0] if len(trials_A) > 0 else 62
                        filters_dict[folder_num][subject][label][band_name] = np.eye(dummy_channels)[:n_components, :]
                        continue
                    covA = compute_average_covariance(trials_A)
                    covB = compute_average_covariance(trials_B)
                    W = compute_csp_filters(covA, covB, n_components=n_components)
                    filters_dict[folder_num][subject][label][band_name] = W
    
    # --- 3단계: 각 trial에 대해 CSP 필터 적용 및 결과 저장 ---
    for folder_num in range(1, 4):
        folder_name = f'{folder_num}_npy_sample'
        folder_path = os.path.join(raw_data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if not filename.endswith('.npy'):
                continue
            subject, sample, label = parse_file_info(filename)
            file_path = os.path.join(folder_path, filename)
            try:
                data = np.load(file_path)  # shape: (channels, T)
            except Exception as e:
                print(f"파일 로드 오류 {file_path}: {e}")
                continue
            band_features = []
            for band_name, (low, high) in bands.items():
                filtered = bandpass_filter(data, low, high, fs)  # (channels, T)
                # 해당 subject, label, band에 대한 CSP 필터 추출
                if (subject in filters_dict[folder_num] and 
                    label in filters_dict[folder_num][subject] and 
                    band_name in filters_dict[folder_num][subject][label]):
                    W = filters_dict[folder_num][subject][label][band_name]  # (8, channels)
                else:
                    W = np.eye(data.shape[0])[:n_components, :]
                # 공간 필터 적용: Y = W * filtered -> (8, T)
                Y = np.dot(W, filtered)
                # (T, 8)로 전치
                Y = Y.T
                band_features.append(Y)
            # 4개 주파수 대역 결과를 스택하여 최종 shape: (4, T, 8)
            features = np.stack(band_features, axis=0)
            out_filename = f'folder{folder_num}_subject{subject}_sample{sample}_label{label}.npy'
            out_path = os.path.join(output_root, out_filename)
            np.save(out_path, features)
            print(f"저장 완료: {out_path} (shape: {features.shape})")

if __name__ == '__main__':
    main()
