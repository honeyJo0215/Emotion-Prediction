import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 작업 디렉토리 기준으로 Face_model.py 를 import 가능하도록
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from Face_model import build_face_model

# ───────────────────────────────────────────────────────────────
# 0) 경로 및 설정
# ───────────────────────────────────────────────────────────────
FEATURE_ROOT   = "/home/bcml1/sigenv/_7월/_rppg/minipatch_rppg_csp"
LABEL_ROOT     = "/home/bcml1/2025_EMOTION/DEAP_5labels"
RESULT_DIR     = "rppg_results"
MODEL_DIR      = "rppg_model"

NUM_SUBJECTS   = 22
NUM_SAMPLES    = 40
TARGET_LEN     = 3000    # CSP-RPPG 신호 길이
N_COMPONENTS   = 10      # CSP 컴포넌트 수

BATCH_SIZE     = 8
EPOCHS         = 200

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────
# 1) 저장된 CSP-RPPG 불러와서 세그먼트(=trial)별 데이터 준비
# ───────────────────────────────────────────────────────────────
def prepare_rppg_segments(feature_root, label_root, num_subj, num_samp):
    X_list, y_list, meta = [], [], []
    for subj in range(1, num_subj+1):
        sid = f"s{subj:02d}"
        labels = np.load(os.path.join(label_root, f"subject{subj:02d}.npy")).astype(int)
        print(f"[DEBUG] Loaded labels for {sid}: {labels.shape}, unique={np.unique(labels)}")
        feat_dir = os.path.join(feature_root, sid)
        for trial in range(1, num_samp+1):
            raw = labels[trial-1]
            if raw == 4:
                continue
            fname = f"trial{trial:02d}_csp_rppg.npy"
            path  = os.path.join(feat_dir, fname)
            if not os.path.isfile(path):
                print(f"[WARN] Missing {path}")
                continue
            comps = np.load(path)    # shape (10,3000)
            # (10,3000) → (3000,10)
            seq   = comps.T
            if seq.shape != (TARGET_LEN, N_COMPONENTS):
                print(f"[WARN] Unexpected shape {seq.shape} in {fname}")
                continue
            X_list.append(seq)
            y_list.append(1 if raw >= 3 else 0)
            meta.append((sid, trial))
    X = np.stack(X_list, axis=0)  # (n_trials, 3000, 10)
    y = np.array(y_list, dtype=np.int32)
    print(f"[INFO] Prepared RPPG data: X={X.shape}, y={y.shape}")
    return X, y, meta

# ───────────────────────────────────────────────────────────────
# main 실행부
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1) 데이터 준비
    X, y, meta = prepare_rppg_segments(FEATURE_ROOT, LABEL_ROOT, NUM_SUBJECTS, NUM_SAMPLES)

    # 2) train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[INFO] Split: X_train={X_train.shape}, X_test={X_test.shape}")

    # 3) 모델 빌드
    num_classes = 2
    seq_len     = X_train.shape[1]  # 3000
    feat_dim    = X_train.shape[2]  # 10
    model = build_face_model(
        num_classes=num_classes,
        seq_length=seq_len,
        feature_dim=feat_dim
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 4) 콜백
    cp = ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_rppg.weights.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    es = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )

    # 5) 학습
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cp, es],
        verbose=2
    )

    # 6) 평가 및 리포트
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {loss:.4f}, Acc: {acc:.4f}")
    y_pred = model.predict(X_test).argmax(axis=1)
    report = classification_report(y_test, y_pred, target_names=['low','high'])
    cm     = confusion_matrix(y_test, y_pred)
    print(report)
    print("Confusion matrix:\n", cm)

    # 7) 결과 저장
    with open(os.path.join(RESULT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    np.savetxt(os.path.join(RESULT_DIR, 'confusion_matrix.txt'), cm, fmt='%d')

    # 8) 학습곡선 저장
    plt.figure(); plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss'); plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, 'loss_curve.png')); plt.close()

    plt.figure(); plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc'); plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, 'accuracy_curve.png')); plt.close()
