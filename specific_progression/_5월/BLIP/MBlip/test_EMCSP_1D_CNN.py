import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses, metrics
from sklearn.model_selection import train_test_split
from EMCSP_1D_CNN import EMCSP_EEG_1DCNN_Encoder
from collections import defaultdict

def limit_gpu_memory(memory_limit_mib=10000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
            )
            print(f"GPU memory limited to {memory_limit_mib} MiB.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU.")

limit_gpu_memory(10000)

mat_dirs = [
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/1",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/2",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/3",
]
session_labels_list = [
    [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
]

# 1) 전체 raw trials 로드
encoder = EMCSP_EEG_1DCNN_Encoder(apply_smoothing=True, window_len=200)
all_raw = []
for d, labels in zip(mat_dirs, session_labels_list):
    raw = encoder.load_raw_trials(d, labels)
    all_raw.extend(raw)

# 2) 피험자별 trials 분리
subject_trials = defaultdict(list)
for tr in all_raw:
    subject_trials[tr['subject']].append(tr)

# 3) 피험자별 intra-subject 학습/평가
results = {}
for subj, trials in subject_trials.items():
    print(f"--- Subject {subj} ---")
    # 레이블 획득
    y = [tr['label'] for tr in trials]
    # train/test split
    train_tr, test_tr = train_test_split(
        trials, test_size=0.2, stratify=y, random_state=42
    )
    # CSP 필터 학습
    subj_encoder = EMCSP_EEG_1DCNN_Encoder(apply_smoothing=True, window_len=200)
    subj_encoder.compute_filters_from_trials(train_tr)
    # 특성 추출
    X_tr, y_tr = subj_encoder.extract_features_from_trials(train_tr)
    X_te, y_te = subj_encoder.extract_features_from_trials(test_tr)
    # seq_len dim 추가
    X_tr = X_tr[:, np.newaxis, ...]
    X_te = X_te[:, np.newaxis, ...]

    # 모델 구성
    seq_len = X_tr.shape[1]
    num_cls = len(np.unique(y_tr))
    inp = layers.Input(shape=(seq_len, subj_encoder.n_bands, subj_encoder.window_len, subj_encoder.n_channels))
    emb = subj_encoder(inp)
    flat = layers.Flatten()(emb)
    x = layers.Dense(64, activation='relu')(flat)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_cls, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )

    # 학습 & 평가
    model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=30, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"Subject {subj} — Test Loss: {loss:.4f}, Acc: {acc:.4f}")
    results[subj] = acc

print("\n=== Intra-Subject Results ===")
for subj, accuracy in results.items():
    print(f"Subject {subj}: {accuracy:.4f}")
