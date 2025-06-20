import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, optimizers, losses, metrics
from collections import defaultdict

# ↓↓↓ 하이퍼파라미터 설정 ↓↓↓
MAT_DIRS = [
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/1",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/2",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/3",
]
SESSION_LABELS_LIST = [
    [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
]

MODE = 'intra'            # 'intra' 또는 'inter'
LEAVE_OUT_SUBJECT = None  # inter 모드일 때 특정 피험자만 테스트하려면 'S01' 등으로 지정

# 데이터 전처리
FS = 200
WINDOW_LEN = 200
APPLY_SMOOTHING = True

# split & 학습
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 모델 트레이닝
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# GPU 메모리 제한 (MiB)
MEMORY_LIMIT_MIB = 10000

#CSP vector
N_COMPONENTS = 8

# ↑↑↑ 하이퍼파라미터 설정 ↑↑↑


# ——————————————————————————————————————————————
# 유틸: GPU 메모리 제한
def limit_gpu_memory(memory_limit_mib=MEMORY_LIMIT_MIB):
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


# 분리된 모듈에서 가져오기
from EMCSP_1D_CNN import EEGDataLoader, EMCSP_EEG_1DCNN_Encoder

# 1) 환경 설정
limit_gpu_memory()

loader = EEGDataLoader(
    fs=FS,
    apply_smoothing=APPLY_SMOOTHING,
    window_len=WINDOW_LEN
)



# — intra‐subject 모드 —
if MODE == 'intra':
    # 1) 전체 세션(Subject, Session) 불러오기
    all_sessions = []
    for mat_dir, sess_labels in zip(MAT_DIRS, SESSION_LABELS_LIST):
        all_sessions.extend(loader.load_raw_sessions(mat_dir, sess_labels))

    results = {}

    # 2) Subject 별 세션 그룹핑
    subjects = sorted({s['subject'] for s in all_sessions})
    for subj in subjects:
        print(f"\n--- [INTRA] Subject {subj} ---")

        # 3) 이 피험자(subject)만 필터링
        subj_sessions = [s for s in all_sessions if s['subject'] == subj]

        # 4) 세션 단위로 train/test split
        sess_tr, sess_te = loader.split_trials(
            subj_sessions,
            mode='intra',
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        print("  ▶ Train sessions :", [s['session'] for s in sess_tr])
        print("   ▶  Test sessions :", [s['session'] for s in sess_te])

        # 5) 윈도우(trial) 단위로 세분화
        train_tr = loader.segment_sessions(sess_tr)
        test_tr  = loader.segment_sessions(sess_te)
        print(f"  ▶ Train-window trials: {len(train_tr)}")
        print(f"   ▶  Test-window trials: {len(test_tr)}")

        train_tr2, val_tr = train_test_split(
        train_tr,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=[t['label'] for t in train_tr]
    )
        # 6) CSP 필터 학습
        enc = EMCSP_EEG_1DCNN_Encoder(
            fs=FS,
            window_len=WINDOW_LEN,
            apply_smoothing=APPLY_SMOOTHING,
            n_channels=N_COMPONENTS
        )
        enc.compute_filters_from_trials(train_tr)

        # 7) 특성 추출
        X_tr, y_tr = enc.extract_features_from_trials(train_tr2)
        X_val, y_val = enc.extract_features_from_trials(val_tr)
        X_te, y_te = enc.extract_features_from_trials(test_tr)
        X_tr = X_tr[:, np.newaxis, ...]
        X_val = X_val[:, np.newaxis, ...]
        X_te = X_te[:, np.newaxis, ...]

        # 8) 모델 구성
        seq_len = 1
        num_cls = len(np.unique(y_tr))
        inp = layers.Input(shape=(seq_len, enc.n_bands, WINDOW_LEN, enc.n_components))
        emb = enc(inp)
        flat = layers.Flatten()(emb)
        x = layers.Dense(64, activation='relu')(flat)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(num_cls, activation='softmax')(x)
        model = Model(inp, out)
        model.compile(
            optimizer=optimizers.Adam(LEARNING_RATE),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[metrics.SparseCategoricalAccuracy()]
        )

        # 9) 학습 & 평가
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        loss, acc = model.evaluate(X_te, y_te, verbose=1)
        print(f"  Subject {subj} — Test Loss: {loss:.4f}, Acc: {acc:.4f}")
        results[subj] = acc

    # 10) Summary
    print("\n=== INTRA-Subject Results ===")
    for subj, acc in results.items():
        print(f"  Subject {subj}: {acc:.4f}")
# — inter‐subject 모드 (leave-one-out) —
elif MODE == 'inter':
    all_trials = []
    for mat_dir, sess_labels in zip(MAT_DIRS, SESSION_LABELS_LIST):
        trials = loader.load_raw_trials(mat_dir, sess_labels)
        all_trials.extend(trials)

    subject_trials = defaultdict(list)
    for tr in all_trials:
        subject_trials[tr['subject']].append(tr)

    results = {}
    # 테스트할 피험자 목록 정하기
    subjects = ([LEAVE_OUT_SUBJECT]
                if LEAVE_OUT_SUBJECT
                else sorted(subject_trials.keys()))

    for subj in subjects:
        print(f"\n--- [INTER] Leave‐Out Subject {subj} ---")
        train_tr, test_tr = loader.split_trials(
            all_trials,
            mode='inter',
            leave_out_subject=subj
        )
        
        enc = EMCSP_EEG_1DCNN_Encoder(
            fs=FS,
            window_len=WINDOW_LEN,
            apply_smoothing=APPLY_SMOOTHING,
            n_channels=N_COMPONENTS
        )
        enc.compute_filters_from_trials(train_tr)

        X_tr, y_tr = enc.extract_features_from_trials(train_tr)
        X_te, y_te = enc.extract_features_from_trials(test_tr)
        X_tr = X_tr[:, np.newaxis, ...]
        X_te = X_te[:, np.newaxis, ...]

        seq_len = 1
        num_cls = len(np.unique(y_tr))
        inp = layers.Input(shape=(seq_len, enc.n_bands, WINDOW_LEN, enc.n_components))
        emb = enc(inp)
        flat = layers.Flatten()(emb)
        x = layers.Dense(64, activation='relu')(flat)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(num_cls, activation='softmax')(x)
        model = Model(inp, out)
        model.compile(
            optimizer=optimizers.Adam(LEARNING_RATE),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[metrics.SparseCategoricalAccuracy()]
        )

        model.fit(X_tr, y_tr,
                  validation_data=(X_te, y_te),
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  verbose=1)
        loss, acc = model.evaluate(X_te, y_te, verbose=1)
        print(f"Leave‐Out {subj} — Test Loss: {loss:.4f}, Acc: {acc:.4f}")
        results[subj] = acc

    print("\n=== INTER-Subject (Leave-One-Out) Results ===")
    for subj, acc in results.items():
        print(f"  Left‐Out {subj}: {acc:.4f}")

else:
    raise ValueError("MODE는 'intra' 또는 'inter' 중 하나로 설정해야 합니다.")
