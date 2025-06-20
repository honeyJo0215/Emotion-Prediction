import os, re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
def load_csp_features(csp_dir):
    """
    csp_dir 내 .npy 파일들을 읽어,
    각 파일명에 포함된 subject, label을 추출하고,
    데이터를 1초(200샘플) 단위로 분할해 X, Y, subjects에 추가.
    반환:
      X: np.ndarray, shape (n_windows, 4,200,8)
      Y: np.ndarray, shape (n_windows,)
      S: np.ndarray, shape (n_windows,) – subject ID 문자열
    """
    Xs, Ys, Ss = [], [], []
    for fn in os.listdir(csp_dir):
        if not fn.endswith(".npy"):
            continue
        arr = np.load(os.path.join(csp_dir, fn))
        if arr.ndim != 3 or arr.shape[0] != 4 or arr.shape[2] != 8:
            continue
        sm = re.search(r'subject(\d+)', fn, re.IGNORECASE)
        lm = re.search(r'label(\d+)',   fn, re.IGNORECASE)
        if not sm or not lm:
            continue
        subj  = sm.group(1)
        label = int(lm.group(1))
        T     = arr.shape[1]
        n_win = T // 200
        for i in range(n_win):
            win = arr[:, i*200:(i+1)*200, :]
            Xs.append(win)
            Ys.append(label)
            Ss.append(subj)
    X = np.array(Xs, dtype=np.float32)
    Y = np.array(Ys, dtype=np.int32)
    S = np.array(Ss)
    print(f"Loaded {X.shape[0]} windows → labels {np.unique(Y)}, subjects {np.unique(S)}")
    return X, Y, S

# =============================================================================
if __name__ == "__main__":
    CSP_DIR    = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"
    RESULT_DIR = "/home/bcml1/sigenv/_5월/svm_seed/intra_simple_svm"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1) CSP 데이터 로드
    X, Y, S = load_csp_features(CSP_DIR)
    unique_subjs = sorted(np.unique(S))

    # 2) Subject별 Intra-Subject 학습/평가
    for subj in unique_subjs:
        subj_dir = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_dir, exist_ok=True)

        mask = (S == subj)
        Xs, Ys = X[mask], Y[mask]
        if len(Ys) < 10:
            print(f"[Subject {subj}] 샘플 부족({len(Ys)}), 스킵")
            continue

        # 윈도우 플래튼: (n_samples, 6400)
        X_flat = Xs.reshape(len(Ys), -1)

        # Train/Test 분할 (stratify)
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, Ys, test_size=0.3, random_state=42, stratify=Ys
        )

        # Pipeline 정의
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=0.95, svd_solver="full")),
            ("svc",    SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True
            ))
        ])

        # SVM 학습
        print(f"[Subject {subj}] Training on {len(y_train)} samples")
        pipe.fit(X_train, y_train)

        # 평가
        y_pred = pipe.predict(X_test)
        report = classification_report(y_test, y_pred)
        cm     = confusion_matrix(y_test, y_pred)

        # 2-1) Confusion Matrix 저장
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title(f"Subject {subj} Confusion Matrix")
        cm_path = os.path.join(subj_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # 2-2) Classification Report 저장
        rpt_path = os.path.join(subj_dir, "classification_report.txt")
        with open(rpt_path, "w") as f:
            f.write(report)

        # 2-3) 모델 저장
        model_path = os.path.join(subj_dir, "svm_ecsp_model.joblib")
        joblib.dump(pipe, model_path)

        print(f"[Subject {subj}] Saved CM, report, and model to {subj_dir}")
