#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leave-One-Subject-Out evaluation using DiffuSSM-enhanced CNN on SEED-IV dataset
Inter-subject generalization: train on subjects 1-15 except one, test on the held-out subject.

This version reduces memory usage by casting data to float32, lowering batch size,
using tf.data optimizations, and clearing temporary lists.
"""

import os
# Disable XLA devices to prevent missing libdevice errors
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
# Limit GPU memory
import tensorflow as tf
tf.config.optimizer.set_jit(False)
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.data import AUTOTUNE
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 결과 저장 경로 설정
RESULT_DIR = "/home/bcml1/sigenv/_5월/inter_diffusion/inter_seed_rx1"
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# GPU 메모리 제한 (필요시 사용)
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

limit_gpu_memory(18000)

# =============================================================================
# Layer definitions (unchanged)
class BidirectionalSSMLayer(layers.Layer):
    def __init__(self, units, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.fwd_conv = layers.Conv1D(filters=1, kernel_size=kernel_size,
                                      padding='causal', activation=None)
        self.bwd_conv = layers.Conv1D(filters=1, kernel_size=kernel_size,
                                      padding='causal', activation=None)

    def call(self, x, training=False):
        seq = tf.expand_dims(x, axis=-1)
        fwd = self.fwd_conv(seq)
        rev = tf.reverse(seq, axis=[1])
        bwd = self.bwd_conv(rev)
        bwd = tf.reverse(bwd, axis=[1])
        return tf.squeeze(fwd + bwd, axis=-1)

class DiffuSSMLayer(layers.Layer):
    def __init__(self, model_dim, hidden_dim, cond_dim=None, alpha_init=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization()
        self.cond_mlp = layers.Dense(2 * model_dim) if cond_dim else None
        self.hg_down = layers.Dense(hidden_dim, activation='swish')
        self.ssm = BidirectionalSSMLayer(hidden_dim)
        self.hg_up = layers.Dense(model_dim)
        self.alpha = self.add_weight(name="fusion_scale_alpha", shape=(),
                                     initializer=tf.keras.initializers.Constant(alpha_init),
                                     trainable=True)

    def call(self, x, cond=None, training=False):
        x_ln = self.norm(x)
        if self.cond_mlp and cond is not None:
            gamma, beta = tf.split(self.cond_mlp(cond), 2, axis=-1)
            x_ln = gamma * x_ln + beta
        h = self.hg_down(x_ln)
        h = self.ssm(h)
        h = self.hg_up(h)
        return x + self.alpha * h

# Gaussian noise

def add_diffusion_noise(x, stddev=0.05):
    return x + tf.random.normal(tf.shape(x), stddev=stddev)

# =============================================================================
# CNN Branch Builders (unchanged structure)

def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda x: add_diffusion_noise(x, noise_std))(inp)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    noisy = layers.Lambda(lambda x: add_diffusion_noise(x, noise_std))(x)
    diffu = DiffuSSMLayer(64, 64)(noisy)
    out = layers.Add()([x, diffu])
    return models.Model(inp, out)


def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda x: add_diffusion_noise(x, noise_std))(inp)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    noisy = layers.Lambda(lambda x: add_diffusion_noise(x, noise_std))(x)
    diffu = DiffuSSMLayer(64, 64)(noisy)
    out = layers.Add()([x, diffu])
    return models.Model(inp, out)

# =============================================================================
# Full model builder

def build_separated_model_with_diffusion(num_classes=4, noise_std=0.05):
    inp = layers.Input((4,200,8))
    freq = layers.Permute((2,3,1))(inp)
    freq_out = build_freq_branch_with_diffusion((200,8,4), noise_std)(freq)
    chan = layers.Permute((2,1,3))(inp)
    chan_out = build_chan_branch_with_diffusion((200,4,8), noise_std)(chan)
    x = layers.Concatenate()([freq_out, chan_out])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out)

# =============================================================================
# CSP feature loader (cast to float32, clear lists)

def load_csp_features(csp_dir):
    X_list, Y_list, subj_list = [], [], []
    for fn in os.listdir(csp_dir):
        if not fn.endswith('.npy'): continue
        data = np.load(os.path.join(csp_dir, fn))
        if data.ndim!=3 or data.shape[0]!=4 or data.shape[2]!=8: continue
        T = data.shape[1]
        for i in range(T//200):
            X_list.append(data[:, i*200:(i+1)*200, :])
            Y_list.append(int(re.search(r'label(\d+)', fn, re.IGNORECASE).group(1)))
            subj_list.append(int(re.search(r'subject(\d+)', fn, re.IGNORECASE).group(1)))
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.int8)
    subjects = np.array(subj_list, dtype=np.int8)
    # free memory from lists
    del X_list, Y_list, subj_list
    print(f"Loaded samples: {X.shape[0]}, dtype: {X.dtype}, unique subjects: {np.unique(subjects)}")
    return X, Y, subjects

# =============================================================================
if __name__ == '__main__':
    CSP_DIR = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"
    X, Y, subjects = load_csp_features(CSP_DIR)
    for subj in np.unique(subjects):
        print(f"\n>>> LOSO: test subject={subj}")
        mask_test = subjects==subj
        X_train, Y_train = X[~mask_test], Y[~mask_test]
        X_test , Y_test  = X[mask_test], Y[mask_test]
        if X_test.size==0: continue

        X_tr, X_val, Y_tr, Y_val = train_test_split(
            X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)

        # tf.data pipeline with caching and prefetch
        BATCH = 32
        train_ds = (tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
                    .shuffle(1000).batch(BATCH)
                    .cache().prefetch(AUTOTUNE))
        val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, Y_val))
                    .batch(BATCH).cache().prefetch(AUTOTUNE))
        test_ds  = (tf.data.Dataset.from_tensor_slices((X_test, Y_test))
                    .batch(BATCH).prefetch(AUTOTUNE))

        model = build_separated_model_with_diffusion(num_classes=len(np.unique(Y)), noise_std=0.05)
        opt = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                1e-4, decay_steps=100000, decay_rate=0.9, staircase=True),
            clipnorm=1.0)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        es = EarlyStopping('val_accuracy', patience=50, mode='max', restore_best_weights=True)

        history = model.fit(train_ds, epochs=1000, validation_data=val_ds, callbacks=[es], verbose=1)

        # save curves and results
        out_dir = os.path.join(RESULT_DIR, f"subj_{subj}")
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend(); plt.title('Loss')
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='train acc')
        plt.plot(history.history['val_accuracy'], label='val acc')
        plt.legend(); plt.title('Accuracy')
        plt.savefig(os.path.join(out_dir, 'training_curves.png'))
        plt.close()

        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        preds = np.argmax(model.predict(test_ds), axis=1)
        cm = confusion_matrix(Y_test, preds)
        rep = classification_report(Y_test, preds)

        np.save(os.path.join(out_dir, 'test_acc.npy'), [test_acc])
        with open(os.path.join(out_dir, 'confusion_matrix.txt'), 'w') as f: f.write(np.array2string(cm))
        with open(os.path.join(out_dir, 'report.txt'), 'w') as f: f.write(rep)
        model.save(os.path.join(out_dir, 'model_LOSO.keras'))
        print(f"Saved results for subject {subj}.")
