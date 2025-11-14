import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# ───────────────────────────────────────────────
# 경로 및 하이퍼파라미터 설정
# ───────────────────────────────────────────────
DATA_ROOT   = "/home/bcml1/sigenv/_7월/_rppg/minipatch_rppg_csp"
LABEL_ROOT  = "/home/bcml1/2025_EMOTION/DEAP_5labels"
RESULT_DIR  = "./rppg_cnn_results"
os.makedirs(RESULT_DIR, exist_ok=True)

NUM_CLASSES   = 4
SEGMENT_LEN   = 50      # 프레임 세그먼트 길이
INPUT_CH      = 20      # 채널 수 (patches)
TEST_SIZE     = 0.2     # 테스트 비율
RANDOM_STATE  = 42

# ───────────────────────────────────────────────
# 데이터 로딩 및 세그먼트 분할
# ───────────────────────────────────────────────
X, y, trials = [], [], []

for subj_dir in sorted(os.listdir(DATA_ROOT)):
    subj_id = subj_dir.replace('s', '').zfill(2)
    label_path = os.path.join(LABEL_ROOT, f"subject{subj_id}.npy")
    if not os.path.exists(label_path):
        continue

    labels_all = np.load(label_path).astype(int)
    trial_idx = 0

    for trial_num in range(1, 41):
        fn = f"trial{trial_num:02d}_csp_rppg_multiclass.npy"
        fpath = os.path.join(DATA_ROOT, subj_dir, fn)
        if not os.path.exists(fpath):
            continue
        label = labels_all[trial_idx]
        trial_idx += 1
        if label == 4:
            continue

        data = np.load(fpath)  # (20, 3000)
        n_segs = data.shape[1] // SEGMENT_LEN
        for s in range(n_segs):
            seg = data[:, s*SEGMENT_LEN:(s+1)*SEGMENT_LEN]
            if seg.shape[1] == SEGMENT_LEN:
                X.append(seg.T)
                y.append(label)
                trials.append(f"{subj_id}_trial{trial_num:02d}")

X = np.array(X)       # (samples, SEGMENT_LEN, INPUT_CH)
y = np.array(y)       # (samples,)
trials = np.array(trials)
print(f"[INFO] Loaded segments: {X.shape}, labels: {y.shape}")

# ───────────────────────────────────────────────
# 모델 정의
# ───────────────────────────────────────────────
def build_model():
    model = Sequential([
        Conv1D(64, 5, padding='same', input_shape=(SEGMENT_LEN, INPUT_CH)),
        BatchNormalization(), ReLU(),
        Conv1D(128, 3, padding='same'),
        BatchNormalization(), ReLU(),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ───────────────────────────────────────────────
# 학습 및 평가 함수
# ───────────────────────────────────────────────
def train_and_evaluate(X_tr, X_te, y_tr, y_te, mode):
    print(f"\n[TRAINING MODE] {mode}")
    model = build_model()
    ckpt_path = os.path.join(RESULT_DIR, f"model_{mode}.h5")
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, save_best_only=True)
    ]
    history = model.fit(
        X_tr, y_tr,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # --- 저장: 학습곡선 ---
    plt.figure(); plt.plot(history.history['accuracy'], label='train_acc'); plt.plot(history.history['val_accuracy'], label='val_acc'); plt.legend(); plt.title('Accuracy'); plt.savefig(os.path.join(RESULT_DIR, f'acc_{mode}.png')); plt.close()
    plt.figure(); plt.plot(history.history['loss'], label='train_loss'); plt.plot(history.history['val_loss'], label='val_loss'); plt.legend(); plt.title('Loss'); plt.savefig(os.path.join(RESULT_DIR, f'loss_{mode}.png')); plt.close()

    # --- 저장: Classification Report ---
    y_pred = model.predict(X_te).argmax(axis=1)
    report = classification_report(y_te, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(RESULT_DIR, f'report_{mode}.csv'), index=True)

    # --- 저장: Confusion Matrix ---
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt='d'); plt.title(f'Confusion Matrix ({mode})'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.savefig(os.path.join(RESULT_DIR, f'cm_{mode}.png')); plt.close()

# ───────────────────────────────────────────────
# 분할 방식 1: Trial-wise split (trial 단위 겹치지 않게)
# ───────────────────────────────────────────────
unique_trials = np.unique(trials)
tr_tr, tr_te = train_test_split(unique_trials, test_size=TEST_SIZE, random_state=RANDOM_STATE)
mask_tr = np.isin(trials, tr_tr)
mask_te = np.isin(trials, tr_te)
train_and_evaluate(X[mask_tr], X[mask_te], y[mask_tr], y[mask_te], mode='trial_split')

# ───────────────────────────────────────────────
# 분할 방식 2: Segment-wise split (랜덤 세그먼트)
# ───────────────────────────────────────────────
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
train_and_evaluate(X_tr2, X_te2, y_tr2, y_te2, mode='segment_split')
