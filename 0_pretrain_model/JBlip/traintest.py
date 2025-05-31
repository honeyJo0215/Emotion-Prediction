#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# 사용자 정의 모듈
from components import MultiModalDataLoader
from blip_itm import blip_itm
from blip_nlvr import blip_nlvr

# ── GPU 메모리 제한 ────────────────────────────────────────────────────────
def limit_gpu_memory(memory_limit_mib=10000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
        )
        print(f"GPU memory limited to {memory_limit_mib} MiB.")
    else:
        print("No GPU available, using CPU.")
limit_gpu_memory(17000)

def main():
    # Paths
    video_root = "/home/bcml1/2025_EMOTION/face_video"
    eeg_root   = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label"

    # Hyperparameters
    fs = 200
    window_len = 200
    apply_smoothing = True
    n_components = 8
    frame_size = (480, 480)

    test_size = 0.2
    random_state = 42
    batch_size = 16
    epochs = 10
    learning_rate = 1e-4

    # 데이터 로드
    loader = MultiModalDataLoader(
        video_root=video_root,
        eeg_root=eeg_root,
        fs=fs,
        window_len=window_len,
        apply_smoothing=apply_smoothing,
        n_components=n_components,
        frame_size=frame_size
    )
    X_img, X_seq, y = loader.load()

    # 레이블을 0-based 정수로 매핑
    classes, y_mapped = np.unique(y, return_inverse=True)
    num_classes = len(classes)
    print(f"Loaded {len(y)} samples with {num_classes} classes.")

    # Train/Test 분할
    Xi_tr, Xi_te, Xs_tr, Xs_te, y_tr, y_te = train_test_split(
        X_img, X_seq, y_mapped,
        test_size=test_size,
        stratify=y_mapped,
        random_state=random_state
    )

    # tf.data.Dataset 생성
    def make_ds(Xi, Xs, y):
        ds = tf.data.Dataset.from_tensor_slices(((Xi, Xs), y))
        ds = ds.shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(Xi_tr, Xs_tr, y_tr)
    test_ds  = make_ds(Xi_te, Xs_te, y_te)

    # --- ITM 모델 ---
    print("Initializing ITM model...")
    itm_model = blip_itm(seq_encoder=None, image_size=frame_size[0], embed_dim=256)
    # 분류 헤드 수정
    itm_model.itm_head = layers.Dense(num_classes)
    itm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    print("Training ITM classifier...")
    itm_model.fit(train_ds, validation_data=test_ds, epochs=epochs)

    # --- NLVR 모델 ---
    print("Initializing NLVR model...")
    nlvr_model = blip_nlvr(seq_encoder=None, image_size=frame_size[0], embed_dim=256)
    # 분류 헤드 수정
    nlvr_model.cls_head = tf.keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
    ])
    nlvr_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    print("Training NLVR classifier...")
    nlvr_model.fit(train_ds, validation_data=test_ds, epochs=epochs)

    print("Done.")


if __name__ == '__main__':
    main()
