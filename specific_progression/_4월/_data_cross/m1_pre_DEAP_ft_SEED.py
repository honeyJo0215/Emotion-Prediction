# DEAP dataset과 SEEDIV데이터 pretraining과 fine tuning할 때 크로스해서 사용
# (pre: DEAP & ft: SEED | pre: SEED & ft: DEAP)
# 또한 fine tuning 데이터와 테스트 데이터를 분리해서 진행하기. 4:6으로 분리하기.
# (pre: DEAP & ft: SEED(4) & test: SEED(6) | pre: SEED & ft: DEAP(4) & DEAP(6))
# 단, SEED data는 DEAP과 겹치는 32개 채널만 사용하기 -> 그중 가장 높은 4개 채널, 가장 낮은 4개채널로 총 8개의 채널을 사용
# fine tuning은 intra로 학습하는것이라고 교수님께서 말씀하심

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# GPU 메모리 제한 설정
# =============================================================================
def limit_gpu_memory(memory_limit_mib=5000):
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
limit_gpu_memory(5000)

# =============================================================================
# DiffuSSM Layer
# =============================================================================
class DiffuSSMLayer(layers.Layer):
    def __init__(self, hidden_dim=64, output_units=None, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.norm1 = layers.LayerNormalization()
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.norm2 = layers.LayerNormalization()
        final_units = output_units if output_units is not None else hidden_dim
        self.out_dense = layers.Dense(final_units)
        self.norm_out = layers.LayerNormalization()

    def call(self, x, training=False):
        h = self.norm1(self.dense1(x))
        h = self.norm2(self.dense2(h))
        return self.norm_out(self.out_dense(h))

# =============================================================================
# Noise addition
# =============================================================================
def add_diffusion_noise(x, stddev=0.05):
    return x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)

# =============================================================================
# Frequency branch
# =============================================================================
def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    noisy = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(x)
    diff = DiffuSSMLayer(hidden_dim=64)(noisy)
    out = layers.Add()([x, diff])
    return models.Model(inputs=inp, outputs=out, name='FreqBranchDiff')

# =============================================================================
# Channel branch
# =============================================================================
def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    noisy = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(x)
    diff = DiffuSSMLayer(hidden_dim=64)(noisy)
    out = layers.Add()([x, diff])
    return models.Model(inputs=inp, outputs=out, name='ChanBranchDiff')

# =============================================================================
# Full model
# =============================================================================
def build_separated_model_with_diffusion(num_classes=4, noise_std=0.05):
    inp = layers.Input(shape=(4,200,8), name='CSP_Input')
    f = layers.Permute((2,3,1))(inp)
    freq_branch = build_freq_branch_with_diffusion(noise_std=noise_std)
    f_feat = freq_branch(f)
    c = layers.Permute((2,1,3))(inp)
    chan_branch = build_chan_branch_with_diffusion(noise_std=noise_std)
    c_feat = chan_branch(c)
    x = layers.Concatenate()([f_feat, c_feat])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inp, outputs=out, name='Separated_CNN_Model_with_Diffusion')

# =============================================================================
# 1) DEAP 전용 CSP feature 로더
# =============================================================================
def load_deap_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"):
    X_list, Y_list, subjects_list, files_list = [], [], [], []
    for fn in os.listdir(csp_feature_dir):
        if not fn.endswith(".npy"): continue
        data = np.load(os.path.join(csp_feature_dir, fn))   # (4, T, 8)
        # label, subject, 윈도우 분할
        label_m = re.search(r'label(\d+)', fn, re.IGNORECASE)
        subj_m  = re.search(r'subject(\d+)', fn, re.IGNORECASE)
        if not label_m or not subj_m: continue
        label, subj = int(label_m.group(1)), subj_m.group(1)
        T = data.shape[1]; n_win = T // 200
        for i in range(n_win):
            X_list.append(data[:, i*200:(i+1)*200, :])
            Y_list.append(label)
            subjects_list.append(subj)
            files_list.append(f"{fn}_win{i}")
    X = np.array(X_list); Y = np.array(Y_list)
    subjects = np.array(subjects_list); file_ids = np.array(files_list)
    return X, Y, subjects, file_ids

# =============================================================================
# 2) SEED 전용 CSP feature 로더
# =============================================================================
def load_seed_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"):
    X_list, Y_list, subjects_list, files_list = [], [], [], []
    for fn in os.listdir(csp_feature_dir):
        if not fn.endswith(".npy"): continue
        data = np.load(os.path.join(csp_feature_dir, fn))   # (4, T, 8)
        subj_m  = re.search(r'subject(\d+)', fn, re.IGNORECASE)
        label_m = re.search(r'label(\d+)',   fn, re.IGNORECASE)
        if not subj_m or not label_m: continue
        subj, label = subj_m.group(1), int(label_m.group(1))
        T = data.shape[1]; n_win = T // 200
        for i in range(n_win):
            X_list.append(data[:, i*200:(i+1)*200, :])
            Y_list.append(label)
            subjects_list.append(subj)
            files_list.append(f"{fn}_win{i}")
    X = np.array(X_list); Y = np.array(Y_list)
    subjects = np.array(subjects_list); file_ids = np.array(files_list)
    return X, Y, subjects, file_ids

# =============================================================================
# 3) Cross-dataset 실험 루프
# =============================================================================
experiments = [
    # (pretrain용 디렉토리, finetune용 디렉토리, subject id 리스트,
    #   preloader, ftloader, 이름)
    (
      "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP",
      "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP",
      list(range(1,16)),
      load_deap_csp_features,
      load_seed_csp_features,
      "DEAP2SEED"
    ),
    (
      "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP",
      "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP",
      list(range(1,33)),
      load_seed_csp_features,
      load_deap_csp_features,
      "SEED2DEAP"
    )
]

# for src_dir, tgt_dir, subs, loader_src, loader_tgt, exp_name in experiments:
#     # 1) Pretraining: 80% train / 20% val
#     Xs, ys, _, _ = load_deap_csp_features(csp_feature_dir = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP")   # or load_seed_csp_features
# Xs = Xs.astype('float32')
# ys = ys.astype('int32')

# # 80:20 분리 (오직 Xs, ys)
# X_train_pre, X_val_pre, y_train_pre, y_val_pre = train_test_split(
#     Xs, ys,
#     test_size=0.2,
#     stratify=ys,
#     random_state=42
# )

# model = build_separated_model_with_diffusion(num_classes=len(np.unique(ys)))
# model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
# model.fit(
#     x = X_train_pre.astype('float32'),
#     y = y_train_pre.astype('int32'),
#     validation_data=(
#         X_val_pre.astype('float32'),
#         y_val_pre.astype('int32')
#     ),
#     epochs=20,
#     batch_size=32,
#     # /verbose=0
# )
# w_pre = model.get_weights()


# # ----------------------------
# # 2) Fine-tuning
# # ----------------------------
# y_true, y_pred = [], []
# for sub in subs:
#     Xt, yt, sub_ids, _ = load_seed_csp_features(csp_feature_dir= "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP")  # or DEAP 로더
#     # 피실험자별로 마스킹
#     mask = (sub_ids.astype(int) == sub)
#     Xsub, ysub = Xt[mask], yt[mask]

#     # (a) 70% train+val vs 30% test
#     X_train_val, X_test, y_train_val, y_test = train_test_split(
#         Xsub, ysub,
#         test_size=0.3,
#         stratify=ysub,
#         random_state=42
#     )

#     # (b) train+val → train(6/7), val(1/7) 로 재분리 (전체 기준 60:10:30)
#     X_train_ft, X_val_ft, y_train_ft, y_val_ft = train_test_split(
#         X_train_val, y_train_val,
#         test_size=1/7,
#         stratify=y_train_val,
#         random_state=42
#     )

#     ft_model = build_separated_model_with_diffusion(num_classes=len(np.unique(ys)))
#     ft_model.set_weights(w_pre)
#     ft_model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
#     ft_model.fit(
#         X_train_ft.astype('float32'),
#         y_train_ft.astype('int32'),
#         validation_data=(
#             X_val_ft.astype('float32'),
#             y_val_ft.astype('int32')
#         ),
#         epochs=10,
#         batch_size=16,
#         verbose=0
#     )

#     preds = ft_model.predict(X_test.astype('float32')).argmax(axis=1)
#     y_true.extend(y_test.tolist())
#     y_pred.extend(preds.tolist())

#     # 결과 저장
#     with open(f"{exp_name}_report.txt","w") as f:
#         f.write(classification_report(y_true, y_pred))
#     cm = confusion_matrix(y_true, y_pred)
#     sns.heatmap(cm, annot=True, fmt="d")
#     plt.title(f"{exp_name} Confusion Matrix")
#     plt.savefig(f"{exp_name}_cm.png")
#     plt.clf()

#     print(f"{exp_name} done")

for src_dir, tgt_dir, subs, loader_src, loader_tgt, exp_name in experiments:
    # 1) Pretraining (수정된 부분)
    Xs, ys, _, _ = loader_src(src_dir)   # ✅ loader_src로 변경!
    Xs = Xs.astype('float32')
    ys = ys.astype('int32')

    X_train_pre, X_val_pre, y_train_pre, y_val_pre = train_test_split(
        Xs, ys, test_size=0.2, stratify=ys, random_state=42
    )

    model = build_separated_model_with_diffusion(num_classes=len(np.unique(ys)))
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    model.fit(
        X_train_pre, y_train_pre,
        validation_data=(X_val_pre, y_val_pre),
        epochs=20, batch_size=32
    )
    w_pre = model.get_weights()

    # 2) Fine-tuning (수정된 부분)
    y_true, y_pred = [], []
    for sub in subs:
        Xt, yt, sub_ids, _ = loader_tgt(tgt_dir)   # ✅ loader_tgt로 변경!
        mask = (sub_ids.astype(int) == sub)
        Xsub, ysub = Xt[mask], yt[mask]

        # train+val(70%) vs test(30%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            Xsub, ysub, test_size=0.3, stratify=ysub, random_state=42
        )

        # train+val → train(60%) : val(10%)
        X_train_ft, X_val_ft, y_train_ft, y_val_ft = train_test_split(
            X_train_val, y_train_val,
            test_size=1/7, stratify=y_train_val, random_state=42
        )

        ft_model = build_separated_model_with_diffusion(num_classes=len(np.unique(ys)))
        ft_model.set_weights(w_pre)
        ft_model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
        ft_model.fit(
            X_train_ft, y_train_ft,
            validation_data=(X_val_ft, y_val_ft),
            epochs=10, batch_size=16, verbose=0
        )

        preds = ft_model.predict(X_test).argmax(axis=1)
        y_true.extend(y_test.tolist())
        y_pred.extend(preds.tolist())

    # 결과 저장 (이 부분도 들여쓰기 수정: 피실험자 loop 밖으로 빼내기)
    with open(f"{exp_name}_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{exp_name} Confusion Matrix")
    plt.savefig(f"{exp_name}_cm.png")
    plt.clf()

    print(f"{exp_name} done")
