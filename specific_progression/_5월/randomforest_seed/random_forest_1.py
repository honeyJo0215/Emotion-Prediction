import os
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from copy import deepcopy
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
def load_csp_features(csp_dir):
    """
    csp_dir 내 .npy 파일을 읽어,
    각 파일명에 포함된 subject, label을 추출하고,
    데이터를 1초(200샘플) 단위로 분할해 X, Y, subjects, file_ids에 추가.
    반환:
      X: np.ndarray, shape (n_windows, 4,200,8)
      Y: np.ndarray, shape (n_windows,)
      S: np.ndarray, shape (n_windows,) – subject ID 문자열
      F: np.ndarray, shape (n_windows,) – file window ID 문자열
    """
    Xs, Ys, Ss, Fs = [], [], [], []
    for fn in os.listdir(csp_dir):
        if not fn.endswith(".npy"):
            continue
        arr = np.load(os.path.join(csp_dir, fn))
        if arr.ndim!=3 or arr.shape[0]!=4 or arr.shape[2]!=8:
            continue
        m_sub = re.search(r'subject(\d+)', fn, re.IGNORECASE)
        m_lab = re.search(r'label(\d+)',   fn, re.IGNORECASE)
        if not m_sub or not m_lab:
            continue
        subj  = m_sub.group(1)
        label = int(m_lab.group(1))
        T = arr.shape[1]
        n_win = T // 200
        for i in range(n_win):
            win_id = f"{fn}_win{i}"
            win = arr[:, i*200:(i+1)*200, :]
            Xs.append(win)
            Ys.append(label)
            Ss.append(subj)
            Fs.append(win_id)
    return (np.array(Xs, dtype=np.float32),
            np.array(Ys, dtype=np.int32),
            np.array(Ss, dtype=str),
            np.array(Fs, dtype=str))

# =============================================================================
if __name__ == "__main__":
    CSP_DIR    = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"
    RESULT_DIR = "/home/bcml1/sigenv/_5월/randomforest_seed/random_forest_1"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1) CSP 데이터 로드
    X, Y, S, F = load_csp_features(CSP_DIR)
    subjects = sorted(np.unique(S))

    # 2) Subject별 Intra-Subject 학습/평가
    for subj in subjects:
        subj_dir = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_dir, exist_ok=True)

        # 해당 subject 샘플 선택
        mask = (S == subj)
        X_subj = X[mask]
        Y_subj = Y[mask]
        F_subj = F[mask]
        files  = np.unique(F_subj)
        if len(files) < 3:
            print(f"[Subject {subj}] 파일 수 부족({len(files)}개), 스킵")
            continue

        # 파일 단위 train/val/test split
        train_files, test_files = train_test_split(files, test_size=0.3,
                                                   random_state=42)
        train_files, val_files  = train_test_split(train_files, test_size=0.25,
                                                   random_state=42)

        # 마스크 생성
        tr_mask = np.isin(F_subj, train_files)
        va_mask = np.isin(F_subj, val_files)
        te_mask = np.isin(F_subj, test_files)

        X_tr_win, Y_tr = X_subj[tr_mask], Y_subj[tr_mask]
        X_va_win, Y_va = X_subj[va_mask], Y_subj[va_mask]
        X_te_win, Y_te = X_subj[te_mask], Y_subj[te_mask]

        # 윈도우 플래튼
        X_tr = X_tr_win.reshape(len(Y_tr), -1)
        X_va = X_va_win.reshape(len(Y_va), -1)
        X_te = X_te_win.reshape(len(Y_te), -1)

        # 3) 스케일링 및 PCA
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        pca = PCA(n_components=0.95, svd_solver='full').fit(X_tr_s)
        X_tr_p = pca.transform(X_tr_s)
        X_va_p = pca.transform(X_va_s)
        X_te_p = pca.transform(X_te_s)

        # 스케일러/PCA 저장
        joblib.dump(scaler, os.path.join(subj_dir, "scaler.joblib"))
        joblib.dump(pca,    os.path.join(subj_dir, "pca.joblib"))

        # 4) Random Forest warm_start 학습 곡선
        max_trees = 100
        step      = 10
        forest    = RandomForestClassifier(
                        warm_start=True,
                        n_jobs=-1,
                        random_state=42
                    )
        train_acc = []
        val_acc   = []
        best_val  = 0.0
        best_rf   = None

        for n in range(step, max_trees+1, step):
            forest.n_estimators = n
            forest.fit(X_tr_p, Y_tr)
            tr_score = forest.score(X_tr_p, Y_tr)
            va_score = forest.score(X_va_p, Y_va)
            train_acc.append(tr_score)
            val_acc.append(va_score)
            if va_score > best_val:
                best_val = va_score
                best_rf  = deepcopy(forest)

        # 학습 곡선 저장
        plt.figure(figsize=(8,5))
        trees = list(range(step, max_trees+1, step))
        plt.plot(trees, train_acc, label="Train Acc")
        plt.plot(trees, val_acc,   label="Val Acc")
        plt.xlabel("Number of Trees")
        plt.ylabel("Accuracy")
        plt.title(f"RF Learning Curve (Subject {subj})")
        plt.legend()
        curve_path = os.path.join(subj_dir, "training_curves.png")
        plt.savefig(curve_path)
        plt.close()

        # 5) 최적 모델로 테스트 평가
        y_pred = best_rf.predict(X_te_p)
        report = classification_report(Y_te, y_pred)
        cm     = confusion_matrix(Y_te, y_pred)

        # 혼동행렬 저장
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title(f"Subject {subj} Confusion Matrix")
        cm_path = os.path.join(subj_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # Classification Report 저장
        rpt_path = os.path.join(subj_dir, "classification_report.txt")
        with open(rpt_path, "w") as f:
            f.write(report)

        # 6) 모델 저장
        model_path = os.path.join(subj_dir, "rf_ecsp_model.joblib")
        joblib.dump(best_rf, model_path)

        print(f"[Subject {subj}] 완료: 모델, CM, 리포트 저장됨")
