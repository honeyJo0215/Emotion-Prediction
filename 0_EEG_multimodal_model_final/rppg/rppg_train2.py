import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# ───────────────────────────────────────────────
# 경로 설정
# ───────────────────────────────────────────────
DATA_ROOT = "/home/bcml1/sigenv/_7월/_rppg/minipatch_rppg_csp"
LABEL_ROOT = "/home/bcml1/2025_EMOTION/DEAP_5labels"
RESULT_DIR = "./rppg_cnn_results"
os.makedirs(RESULT_DIR, exist_ok=True)

NUM_CLASSES = 4
INPUT_LEN = 3000
INPUT_CH = 20  # 4 classes × 5 CSP filters

# ───────────────────────────────────────────────
# 데이터 로딩
# ───────────────────────────────────────────────
X, y, groups = [], [], []

for subj_dir in sorted(os.listdir(DATA_ROOT)):
    subj_path = os.path.join(DATA_ROOT, subj_dir)
    subj_id = subj_dir.replace("s", "").zfill(2)
    label_path = os.path.join(LABEL_ROOT, f"subject{subj_id}.npy")

    if not os.path.exists(label_path):
        print(f"[WARN] Missing label: {label_path}")
        continue

    labels_all = np.load(label_path).astype(int)  # (40,)
    trial_idx = 0

    for trial_num in range(1, 41):
        fname = f"trial{trial_num:02d}_csp_rppg_multiclass.npy"
        fpath = os.path.join(subj_path, fname)
        if not os.path.exists(fpath):
            continue  # 라벨 4였던 경우 저장 안됨

        label = labels_all[trial_idx]
        trial_idx += 1
        if label == 4:
            continue  # 사용 안 함

        data = np.load(fpath)  # (20, 3000)
        X.append(data.T)  # (3000, 20)
        y.append(label)
        groups.append(subj_id)

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

print(f"[INFO] Loaded data shape: {X.shape}, Labels: {y.shape}, Subjects: {np.unique(groups)}")

# ───────────────────────────────────────────────
# 모델 정의
# ───────────────────────────────────────────────
def build_model():
    model = Sequential([
        Conv1D(64, 5, padding='same', input_shape=(INPUT_LEN, INPUT_CH)),
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
# Inter-subject GroupKFold
# ───────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n[INFO] Fold {fold+1}")
    x_train, x_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = build_model()
    ckpt_path = os.path.join(RESULT_DIR, f"model_fold{fold+1}.h5")
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, save_best_only=True)
    ]
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # ───────────────────────
    # 결과 저장
    # ───────────────────────
    # 학습 곡선
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(os.path.join(RESULT_DIR, f"acc_curve_fold{fold+1}.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(RESULT_DIR, f"loss_curve_fold{fold+1}.png"))
    plt.close()

    # classification report & confusion matrix
    y_pred = model.predict(x_test).argmax(axis=1)
    report = classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(NUM_CLASSES)], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(RESULT_DIR, f"classification_report_fold{fold+1}.csv"))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix Fold {fold+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(RESULT_DIR, f"confusion_matrix_fold{fold+1}.png"))
    plt.close()
