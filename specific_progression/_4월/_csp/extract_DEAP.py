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

def compute_average_covariance(trials, eps=1e-10):
    """
    여러 trial (각 trial: (channels, T))에 대해 정규화된 평균 공분산 행렬 계산.
    각 trial에 대해 cov = X*X.T / trace(X*X.T)를 계산 후 평균.
    trace가 0에 가까운 경우 eps를 더해 정규화를 방지.
    """
    cov_sum = None
    for X in trials:
        cov = np.dot(X, X.T)
        trace_val = np.trace(cov)
        if trace_val < eps:
            trace_val = eps
        cov = cov / trace_val
        if cov_sum is None:
            cov_sum = cov
        else:
            cov_sum += cov
    avg_cov = cov_sum / len(trials)
    return avg_cov

def compute_csp_filters(covA, covB, n_components=8, eps=1e-10):
    """
    두 클래스(클래스 A와 B)의 평균 공분산 행렬을 이용해 CSP 필터 계산.
    표준 CSP 방법 (whitening transform 및 eigen-decomposition)을 사용하여,
    상위 n_components/2와 하위 n_components/2 (총 n_components)의 필터를 추출함.
    반환 W_selected: (n_components, channels)
    """
    R = covA + covB
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Whitening transform: P = Λ^(-1/2) * U^T (eps로 정규화)
    P = np.dot(np.diag(1.0 / np.sqrt(eigvals + eps)), eigvecs.T)
    S = np.dot(P, np.dot(covA, P.T))
    eigvals_S, eigvecs_S = np.linalg.eigh(S)
    sort_idx = np.argsort(eigvals_S)[::-1]
    eigvals_S = eigvals_S[sort_idx]
    eigvecs_S = eigvecs_S[:, sort_idx]
    # Projection matrix: W = (eigvecs_S.T) * P
    W = np.dot(eigvecs_S.T, P)
    n_half = n_components // 2
    W_selected = np.vstack((W[:n_half, :], W[-n_half:, :]))
    return W_selected  # shape: (n_components, channels)

def parse_file_info(filename):
    """
    파일명 형식: "folder{folder}_subject{subject}_sample_{sample}_label{label}.npy"
    예: "folder1_subject1_sample_01_label3.npy"
    folder, subject, sample, label 값을 추출.
    """
    folder_match = re.match(r'folder(\d+)_', filename)
    folder = folder_match.group(1) if folder_match else 'unknown'
    
    subject_match = re.search(r'subject(\d+)', filename)
    subject = subject_match.group(1) if subject_match else 'unknown'
    
    sample_match = re.search(r'sample_(\d+)', filename)
    sample = sample_match.group(1) if sample_match else 'unknown'
    
    label_match = re.search(r'label(\d+)', filename)
    label = label_match.group(1) if label_match else 'unknown'
    
    return folder, subject, sample, label

def main():
    # 경로 설정
    raw_data_root = '/home/bcml1/2025_EMOTION/DEAP_eeg_new_label'
    output_root = '/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP'
    os.makedirs(output_root, exist_ok=True)
    
    fs = 128  # 샘플링 주파수
    n_components = 8
    # 주파수 대역 설정 (Hz)
    bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 50)
    }
    
    # --- 1단계: 그룹화 ---
    group_data = {}  # 구조: group_data[folder][subject][label][band] = [trial_data, ...]
    
    for filename in os.listdir(raw_data_root):
        if not filename.endswith('.npy'):
            continue
        file_path = os.path.join(raw_data_root, filename)
        folder, subject, sample, label = parse_file_info(filename)
        if folder not in group_data:
            group_data[folder] = {}
        if subject not in group_data[folder]:
            group_data[folder][subject] = {}
        if label not in group_data[folder][subject]:
            group_data[folder][subject][label] = {band: [] for band in bands}
        
        try:
            data = np.load(file_path)  # shape: (channels, T)
        except Exception as e:
            print(f"파일 로드 오류 {file_path}: {e}")
            continue
        
        for band_name, (low, high) in bands.items():
            filtered = bandpass_filter(data, low, high, fs)
            group_data[folder][subject][label][band_name].append(filtered)
    
    # --- 2단계: CSP 필터 계산 (One-vs-All) ---
    filters_dict = {}  # 구조: filters_dict[folder][subject][label][band] = W (shape: (8, channels))
    
    for folder in group_data:
        filters_dict[folder] = {}
        for subject in group_data[folder]:
            filters_dict[folder][subject] = {}
            subject_labels = group_data[folder][subject].keys()
            for label in subject_labels:
                filters_dict[folder][subject][label] = {}
                for band_name in bands:
                    trials_A = group_data[folder][subject][label][band_name]
                    trials_B = []
                    for other_label in subject_labels:
                        if other_label == label:
                            continue
                        trials_B.extend(group_data[folder][subject][other_label][band_name])
                    
                    if len(trials_A) == 0 or len(trials_B) == 0:
                        print(f"대상 부족: folder {folder}, subject {subject}, label {label}, band {band_name}")
                        dummy_channels = trials_A[0].shape[0] if len(trials_A) > 0 else 32
                        filters_dict[folder][subject][label][band_name] = np.eye(dummy_channels)[:n_components, :]
                        continue
                    
                    covA = compute_average_covariance(trials_A)
                    covB = compute_average_covariance(trials_B)
                    W = compute_csp_filters(covA, covB, n_components=n_components)
                    filters_dict[folder][subject][label][band_name] = W
    
    # --- 3단계: 각 trial에 대해 CSP 필터 적용 및 결과 저장 ---
    for filename in os.listdir(raw_data_root):
        if not filename.endswith('.npy'):
            continue
        file_path = os.path.join(raw_data_root, filename)
        folder, subject, sample, label = parse_file_info(filename)
        try:
            data = np.load(file_path)  # shape: (channels, T)
        except Exception as e:
            print(f"파일 로드 오류 {file_path}: {e}")
            continue
        
        band_features = []
        for band_name, (low, high) in bands.items():
            filtered = bandpass_filter(data, low, high, fs)
            if (folder in filters_dict and 
                subject in filters_dict[folder] and 
                label in filters_dict[folder][subject] and 
                band_name in filters_dict[folder][subject][label]):
                W = filters_dict[folder][subject][label][band_name]  # (8, channels)
            else:
                W = np.eye(data.shape[0])[:n_components, :]
            # 공간 필터 적용: Y = W * filtered -> (8, T)
            Y = np.dot(W, filtered)
            # (T, 8)로 전치
            Y = Y.T
            band_features.append(Y)
        
        # 4개 주파수 대역 결과를 스택: 최종 shape: (4, T, 8)
        features = np.stack(band_features, axis=0)
        out_filename = f'folder{folder}_subject{subject}_sample{sample}_label{label}.npy'
        out_path = os.path.join(output_root, out_filename)
        np.save(out_path, features)
        print(f"저장 완료: {out_path} (shape: {features.shape})")

if __name__ == '__main__':
    main()
