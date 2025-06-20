"""
DEAP 데이터셋 (EEG, PPS) 기반 멀티모달 감정 분류 모델  
- EEG 데이터: (60, 32, 128) → 파일당 60개의 1초 segment  
- PPS 데이터: (60, 8, 128) → 파일당 60개의 1초 segment  
- 라벨: 각 파일명에 포함 (0: Excited, 1: Relaxed, 2: Stressed, 3: Bored)
- SAMPLE_SECONDS 하이퍼파라미터를 통해 몇 초 단위로 학습할지 결정  
  (예: SAMPLE_SECONDS=3이면, EEG: (32, 128*3, 1), PPS: (8, 128*3, 1))
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # TensorFlow 로그 최소화

import glob
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, Dropout, AveragePooling2D, DepthwiseConv2D,
    LayerNormalization, MultiHeadAttention, Input, Lambda, Add, Concatenate, Softmax
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# GPU memory limit (if required)
def limit_gpu_memory(memory_limit_mib=8000):
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

limit_gpu_memory(8000)


# Hyperparameters and path settings
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-4

# SAMPLE_SECONDS: Decide in seconds to configure the input (1 second = 128 sample)
SAMPLE_SECONDS = 4

# Data path (modified as needed)
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG3"
PPS_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_PPS3"
SAVE_PATH    = "/home/bcml1/sigenv/_3주차_new_EEG+PPS_CNN/4_SEG2s_final1"
os.makedirs(SAVE_PATH, exist_ok=True)

# 1. Dual‑Stream Feature Extractor (EEG & PPS)
def create_dual_stream_feature_extractor(eeg_input_shape=(32,128,1), pps_input_shape=(8,128,1)):
    """
    Feature extraction of EEG and PPS with CNN respectively
    - EEG Branch: Specify input shape (e.g. (32, 128, 1))
    - PPS Branch: Specify input shape (e.g. (8, 128, 1))
    """
    ## EEG Branch
    eeg_input = Input(shape=eeg_input_shape, name="EEG_Input")
    x = Conv2D(16, (4, 16), strides=(2, 8), padding='valid', activation='relu')(eeg_input)
    x = DepthwiseConv2D(kernel_size=(6, 6), strides=(3, 3), padding='valid', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(32, (1, 1), strides=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    eeg_output = Dense(128, activation="relu")(x)
    
    ## PPS Branch
    pps_input = Input(shape=pps_input_shape, name="PPS_Input")
    y = Conv2D(16, (1,16), strides=(1,8), padding='valid', activation='relu')(pps_input)
    y = DepthwiseConv2D(kernel_size=(2,6), strides=(2,3), padding='valid', activation='relu')(y)
    y = AveragePooling2D(pool_size=(2,2), strides=(2,2))(y)
    y = Conv2D(32, (1,1), strides=(1,1), activation='relu')(y)
    y = Flatten()(y)
    y = Dropout(0.25)(y)
    pps_output = Dense(128, activation="relu")(y)
    
    model = Model(inputs=[eeg_input, pps_input], outputs=[eeg_output, pps_output],
                  name="DualStreamFeatureExtractor")
    return model


# 2. Inter‑Modality Fusion Module (Cross‑Modal Transformer)
def create_inter_modality_fusion(eeg_features, pps_features, num_heads=4, d_model=128, dropout_rate=0.25):
    """
    Cross ‑ Modal Transformer to learn the interrelationships between EEG and PPS
    """
    # EEG → PPS
    eeg_query = Dense(d_model)(eeg_features)
    pps_key_value = Dense(d_model)(pps_features)
    eeg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_query)
    pps_key_value = Lambda(lambda x: tf.expand_dims(x, axis=1))(pps_key_value)
    
    cross_att_1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_EEG_to_PPS")(
        query=eeg_query, key=pps_key_value, value=pps_key_value)
    cross_att_1 = Dropout(dropout_rate)(cross_att_1)
    cross_att_1 = Add()([eeg_query, cross_att_1])
    cross_att_1 = LayerNormalization(epsilon=1e-6)(cross_att_1)
    cross_att_1 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_att_1)
    
    # PPS → EEG
    pps_query = Dense(d_model)(pps_features)
    pps_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(pps_query)
    eeg_key_value_2 = Dense(d_model)(eeg_features)
    eeg_key_value_2 = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_key_value_2)
    
    cross_att_2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_PPS_to_EEG")(
        query=pps_query, key=eeg_key_value_2, value=eeg_key_value_2)
    cross_att_2 = Dropout(dropout_rate)(cross_att_2)
    cross_att_2 = Add()([pps_query, cross_att_2])
    cross_att_2 = LayerNormalization(epsilon=1e-6)(cross_att_2)
    cross_att_2 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_att_2)
    
    fused_features = Concatenate(axis=-1)([cross_att_1, cross_att_2])
    fused_features = Dense(d_model, activation="relu", name="Fused_Linear")(fused_features)
    
    # Self-Attention Fusion
    fused_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(fused_features)
    self_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttentionFusion")(
        query=fused_expanded, key=fused_expanded, value=fused_expanded)
    self_att = Dropout(dropout_rate)(self_att)
    self_att = Add()([fused_expanded, self_att])
    self_att = LayerNormalization(epsilon=1e-6)(self_att)
    self_att = Lambda(lambda x: tf.squeeze(x, axis=1))(self_att)
    return self_att

# 3. Intra‑Modality Encoding Module
def create_intra_modality_encoding(eeg_features, pps_features, num_heads=4, d_model=128, dropout_rate=0.25):
    """
    Intra ‑ Modal Encoding preserves and enhances the unique characteristics of EEG and PPS respectively
    """
    # EEG Self-Attention
    eeg_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_features)
    eeg_self_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_EEG")(
        query=eeg_expanded, key=eeg_expanded, value=eeg_expanded)
    eeg_self_att = Dropout(dropout_rate)(eeg_self_att)
    eeg_self_att = Add()([eeg_expanded, eeg_self_att])
    eeg_self_att = LayerNormalization(epsilon=1e-6)(eeg_self_att)
    eeg_self_att = Lambda(lambda x: tf.squeeze(x, axis=1))(eeg_self_att)
    
    # PPS Self-Attention
    pps_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(pps_features)
    pps_self_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_PPS")(
        query=pps_expanded, key=pps_expanded, value=pps_expanded)
    pps_self_att = Dropout(dropout_rate)(pps_self_att)
    pps_self_att = Add()([pps_expanded, pps_self_att])
    pps_self_att = LayerNormalization(epsilon=1e-6)(pps_self_att)
    pps_self_att = Lambda(lambda x: tf.squeeze(x, axis=1))(pps_self_att)
    
    return eeg_self_att, pps_self_att

# 4. Full Model Configuration (EEG, PPS Input → Dual ‑ Stream Extraction → Fusion → Classification)
def build_combined_model(num_classes=4, sample_seconds=1):
    """
    EEG와 PPS 데이터를 받아서  
      - Dual‑Stream Feature Extraction  
      - Cross‑Modal Fusion  
      - Intra‑Modal Encoding  
    을 거쳐 최종 분류 결과를 출력하는 모델 구성  
    sample_seconds: 몇 초 단위로 입력을 구성할지 (예: 3 → time dimension = 128*3)
    """
    time_dim = 128 * sample_seconds
    # Change model input shape
    eeg_input = Input(shape=(32, time_dim, 1), name="EEG_Input")
    pps_input = Input(shape=(8, time_dim, 1), name="PPS_Input")
    
    # The dual extractor takes the input shape as a factor
    dual_extractor = create_dual_stream_feature_extractor(
        eeg_input_shape=(32, time_dim, 1), 
        pps_input_shape=(8, time_dim, 1)
    )
    # Pass new input to dual_extractor model
    eeg_features, pps_features = dual_extractor([eeg_input, pps_input])
    
    fused_features = create_inter_modality_fusion(eeg_features, pps_features)
    eeg_encoded, pps_encoded = create_intra_modality_encoding(eeg_features, pps_features)
    
    inter_classification = Dense(num_classes, activation="softmax", name="Inter_Classification")(fused_features)
    eeg_classification   = Dense(num_classes, activation="softmax", name="EEG_Classification")(eeg_encoded)
    pps_classification   = Dense(num_classes, activation="softmax", name="PPS_Classification")(pps_encoded)
    
    concat_features = Concatenate()([fused_features, eeg_encoded, pps_encoded])
    weights_logits = Dense(units=num_classes, activation=None, name="Weight_Logits")(concat_features)
    weights = Softmax(axis=-1, name="Weight_Softmax")(weights_logits)
    
    model = Model(inputs=[eeg_input, pps_input],
                  outputs=[inter_classification, eeg_classification, pps_classification, weights],
                  name="Multimodal_Emotion_Classifier")
    return model

# 5. Functions for direct loss specification (focal loss) and lambda loss_weights optimization
def get_labels_dict(labels):
    # 모든 출력에 대해 동일한 레이블 (네 번째는 더미 값)
    return {
        "Inter_Classification": labels,
        "EEG_Classification": labels,
        "PPS_Classification": labels,
        "Weight_Softmax": np.zeros((labels.shape[0], 4))
    }

# def grid_search_lambda(train_data, val_data, alpha_values):
#     lambda_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
#     best_acc = 0.0
#     best_lambdas = (0.5, 0.3, 0.2)  # 기본값
    
#     (train_eeg, train_pps), train_labels = train_data
#     (valid_eeg, valid_pps), valid_labels = val_data
#     train_labels_dict = get_labels_dict(train_labels)
#     valid_labels_dict = get_labels_dict(valid_labels)
    
#     for l1, l2, l3 in itertools.product(lambda_grid, repeat=3):
#         if abs(l1 + l2 + l3 - 1.0) > 0.05:
#             continue

#         model = build_combined_model(num_classes=4, sample_seconds=SAMPLE_SECONDS)
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#                       loss={
#                         "Inter_Classification": focal_loss(alpha=alpha_values),
#                         "EEG_Classification": focal_loss(alpha=alpha_values),
#                         "PPS_Classification": focal_loss(alpha=alpha_values),
#                         "Weight_Softmax": lambda y_true, y_pred: tf.constant(0.0)
#                     },
#                     loss_weights={
#                         "Inter_Classification": l1,
#                         "EEG_Classification": l2,
#                         "PPS_Classification": l3,
#                         "Weight_Softmax": 0.0
#                     },
#                     metrics={
#                         "Inter_Classification": "accuracy",
#                         "EEG_Classification": "accuracy",
#                         "PPS_Classification": "accuracy",
#                         "Weight_Softmax": None
#                     }
#                     )
        
#         model.fit([train_eeg, train_pps], train_labels_dict,
#                   validation_data=([valid_eeg, valid_pps], valid_labels_dict),
#                   epochs=3, batch_size=BATCH_SIZE, verbose=2)
#         eval_results = model.evaluate([valid_eeg, valid_pps], valid_labels_dict, verbose=2, return_dict=True)
#         fold_val_acc = eval_results.get("Inter_Classification_accuracy")
#         if fold_val_acc is None:
#             fold_val_acc = eval_results.get("Inter_Classification_sparse_categorical_accuracy")
#         if fold_val_acc is None:
#             print("Available metrics:", eval_results.keys())
#             raise ValueError("Inter Classification accuracy metric not found in evaluation results")
        
#         if fold_val_acc > best_acc:
#             best_acc = fold_val_acc
#             best_lambdas = (l1, l2, l3)

#     print(f" 최적 λ 값 (Grid Search): {best_lambdas}")
#     return best_lambdas

def grid_search_lambda(train_data, val_data, alpha_values):
    lambda_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_acc = 0.0
    best_lambdas = (0.5, 0.3, 0.2)  # Default (updated later)
    
    (train_eeg, train_pps), train_labels = train_data
    (valid_eeg, valid_pps), valid_labels = val_data
    train_labels_dict = get_labels_dict(train_labels)
    valid_labels_dict = get_labels_dict(valid_labels)
    
    # Normalized and enabled for all combinations
    for l1, l2, l3 in itertools.product(lambda_grid, repeat=3):
        continue
        s = l1 + l2 + l3
        l1_norm, l2_norm, l3_norm = l1/s, l2/s, l3/s
        
        model = build_combined_model(num_classes=4, sample_seconds=SAMPLE_SECONDS)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss={
                          "Inter_Classification": focal_loss(alpha=alpha_values),
                          "EEG_Classification": focal_loss(alpha=alpha_values),
                          "PPS_Classification": focal_loss(alpha=alpha_values),
                          "Weight_Softmax": lambda y_true, y_pred: tf.constant(0.0)
                      },
                      loss_weights={
                          "Inter_Classification": l1_norm,
                          "EEG_Classification": l2_norm,
                          "PPS_Classification": l3_norm,
                          "Weight_Softmax": 0.0
                      },
                      metrics={
                          "Inter_Classification": "accuracy",
                          "EEG_Classification": "accuracy",
                          "PPS_Classification": "accuracy",
                          "Weight_Softmax": None
                      })
        
        model.fit([train_eeg, train_pps], train_labels_dict,
                  validation_data=([valid_eeg, valid_pps], valid_labels_dict),
                  epochs=3, batch_size=BATCH_SIZE, verbose=2)
        eval_results = model.evaluate([valid_eeg, valid_pps], valid_labels_dict, verbose=2, return_dict=True)
        fold_val_acc = eval_results.get("Inter_Classification_accuracy")
        if fold_val_acc is None:
            fold_val_acc = eval_results.get("Inter_Classification_sparse_categorical_accuracy")
        if fold_val_acc is None:
            print("Available metrics:", eval_results.keys())
            raise ValueError("Inter Classification accuracy metric not found in evaluation results")
        
        if fold_val_acc > best_acc:
            best_acc = fold_val_acc
            best_lambdas = (l1_norm, l2_norm, l3_norm)
    
    print(f" 최적 λ 값 (Grid Search): {best_lambdas}")
    return best_lambdas


def bayesian_opt_lambda(train_data, val_data, alpha_values, max_iterations=15):
    (train_eeg, train_pps), train_labels = train_data
    (valid_eeg, valid_pps), valid_labels = val_data
    train_labels_dict = get_labels_dict(train_labels)
    valid_labels_dict = get_labels_dict(valid_labels)
    
    def loss_function(lambdas):
        l1, l2, l3 = lambdas
        if abs(l1 + l2 + l3 - 1.0) > 0.05:
            return 1.0

        model = build_combined_model(num_classes=4, sample_seconds=SAMPLE_SECONDS)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={
                "Inter_Classification": "sparse_categorical_crossentropy",
                "EEG_Classification": "sparse_categorical_crossentropy",
                "PPS_Classification": "sparse_categorical_crossentropy",
                "Weight_Softmax": lambda y_true, y_pred: tf.constant(0.0)
            },
            loss_weights={
                "Inter_Classification": np.float32(l1),
                "EEG_Classification": np.float32(l2),
                "PPS_Classification": np.float32(l3),
                "Weight_Softmax": np.float32(0.0)
            },
            metrics={
                "Inter_Classification": "accuracy",
                "EEG_Classification": "accuracy",
                "PPS_Classification": "accuracy",
                "Weight_Softmax": None
            }
        )

        model.fit(
            [train_eeg, train_pps],
            train_labels_dict,
            validation_data=([valid_eeg, valid_pps], valid_labels_dict),
            epochs=3, batch_size=BATCH_SIZE, verbose=2
        )
        eval_results = model.evaluate([valid_eeg, valid_pps], valid_labels_dict, verbose=2, return_dict=True)
        val_acc = eval_results.get("Inter_Classification_accuracy", 
                           eval_results.get("Inter_Classification_sparse_categorical_accuracy"))
        return -val_acc

    result = minimize(
        loss_function, x0=[0.5, 0.3, 0.2],
        bounds=[(0.1, 0.9), (0.1, 0.9), (0.1, 0.9)],
        options={'maxiter': max_iterations}
    )
    best_lambdas = result.x
    print(f" 최적 λ 값 (Bayesian Optimization): {best_lambdas}")
    return best_lambdas

# =============================================================================
# 6. 클래스 빈도 기반 alpha 및 Focal Loss
# =============================================================================
def compute_class_weights(labels, num_classes=4):
    class_counts = np.bincount(labels, minlength=num_classes)
    total = np.sum(class_counts)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = total / (num_classes * class_counts)
    class_weights /= np.max(class_weights)
    return class_weights.astype(np.float32)

def focal_loss(alpha, gamma=2.0):
    def loss(y_true, y_pred):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(alpha))
        alpha_factor = tf.expand_dims(tf.gather(alpha, tf.argmax(y_true_one_hot, axis=-1)), axis=-1)
        loss_val = -alpha_factor * y_true_one_hot * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred + 1e-7)
        return tf.reduce_mean(loss_val)
    return loss

# 7. Data augmentation and balancing functions
# ※ Change factor from 5x to 2x to produce 1 augmented sample for each sample)
def augment_sample(eeg_sample, pps_sample, noise_std_eeg=0.01, noise_std_pps=0.005):
    aug_type = np.random.choice([0, 1])
    if aug_type == 0:
        eeg_aug = eeg_sample + np.random.normal(0, noise_std_eeg, size=eeg_sample.shape)
        pps_aug = pps_sample + np.random.normal(0, noise_std_pps, size=pps_sample.shape)
    else:
        eeg_aug = -eeg_sample
        pps_aug = -pps_sample
    return eeg_aug, pps_aug

def balance_data(train_eeg, train_pps, train_labels):
    unique, counts = np.unique(train_labels, return_counts=True)
    max_count = np.max(counts)
    balanced_eeg, balanced_pps, balanced_labels = [], [], []
    for cls in unique:
        indices = np.where(train_labels == cls)[0]
        if len(indices) < max_count:
            oversampled = np.random.choice(indices, size=max_count, replace=True)
        else:
            oversampled = indices
        balanced_eeg.append(train_eeg[oversampled])
        balanced_pps.append(train_pps[oversampled])
        balanced_labels.append(train_labels[oversampled])
    balanced_eeg = np.concatenate(balanced_eeg, axis=0)
    balanced_pps = np.concatenate(balanced_pps, axis=0)
    balanced_labels = np.concatenate(balanced_labels, axis=0)
    perm = np.random.permutation(balanced_labels.shape[0])
    return balanced_eeg[perm], balanced_pps[perm], balanced_labels[perm]

def augment_all_data(train_eeg, train_pps, train_labels, noise_std_eeg=0.01, noise_std_pps=0.005, factor=2):
    N = train_eeg.shape[0]
    aug_eeg, aug_pps, aug_labels = [], [], []
    for i in range(N):
        e_aug, p_aug = augment_sample(train_eeg[i], train_pps[i], noise_std_eeg, noise_std_pps)
        aug_eeg.append(e_aug)
        aug_pps.append(p_aug)
        aug_labels.append(train_labels[i])
    aug_eeg = np.array(aug_eeg)
    aug_pps = np.array(aug_pps)
    aug_labels = np.array(aug_labels)
    all_eeg = np.concatenate([train_eeg, aug_eeg], axis=0)
    all_pps = np.concatenate([train_pps, aug_pps], axis=0)
    all_labels = np.concatenate([train_labels, aug_labels], axis=0)
    perm = np.random.permutation(all_labels.shape[0])
    return all_eeg[perm], all_pps[perm], all_labels[perm]

# 8. Data Load Function
# Attach all the files for each subject to generate data in the form of (num_files*new_samples, 32, 128*samples_seconds, 1)
def load_multimodal_data(subject, sample_seconds=1):
    """
    파일명 형식: sXX_trial_XX_label_X.npy  
      - EEG 파일 shape: (60, 32, 128)  
      - PPS 파일 shape: (60, 8, 128)  
    각 파일의 60개의 1초 segment를 sample_seconds 단위로 이어붙여 입력 데이터를 구성
    (예: sample_seconds=3이면, 각 파일에서 floor(60/3)=20개의 샘플 생성,
     EEG: (32, 128*3), PPS: (8, 128*3))
    """
    eeg_pattern = os.path.join(EEG_DATA_PATH, f"{subject}_trial_*_label_*.npy")
    pps_pattern = os.path.join(PPS_DATA_PATH, f"{subject}_trial_*_label_*.npy")
    
    eeg_files = sorted(glob.glob(eeg_pattern))
    pps_files = sorted(glob.glob(pps_pattern))
    
    if len(eeg_files) == 0 or len(pps_files) == 0:
        raise FileNotFoundError(f"{subject}에 해당하는 EEG 또는 PPS 파일을 찾을 수 없습니다.")
    
    eeg_list = []
    pps_list = []
    labels_list = []
    
    # Load EEG files and label extraction + segment attachment
    for eeg_file in eeg_files:
        base = os.path.basename(eeg_file)
        parts = base.split('_')
        label_str = parts[-1]  # 예: "0.npy"
        label = int(label_str.split('.')[0])
        eeg_data = np.load(eeg_file)  # shape: (60, 32, 128)
        n_segments = eeg_data.shape[0]
        num_samples = n_segments // sample_seconds
        samples = []
        for i in range(0, num_samples * sample_seconds, sample_seconds):
            # 단순 이어붙이기: axis=-1 (time axis)
            sample = np.concatenate(eeg_data[i:i+sample_seconds], axis=-1)  # shape: (32, 128*sample_seconds)
            samples.append(sample)
        samples = np.stack(samples, axis=0)  # (num_samples, 32, 128*sample_seconds)
        eeg_list.append(samples)
        labels_list.append(np.full((samples.shape[0],), label, dtype=np.int32))
    
    # Load PPS files (assuming the same order as EEG)
    for pps_file in pps_files:
        pps_data = np.load(pps_file)  # shape: (60, 8, 128)
        n_segments = pps_data.shape[0]
        num_samples = n_segments // sample_seconds
        samples = []
        for i in range(0, num_samples * sample_seconds, sample_seconds):
            sample = np.concatenate(pps_data[i:i+sample_seconds], axis=-1)  # (8, 128*sample_seconds)
            samples.append(sample)
        samples = np.stack(samples, axis=0)
        pps_list.append(samples)
    
    eeg_array = np.concatenate(eeg_list, axis=0)      # (total_samples, 32, 128*sample_seconds)
    pps_array = np.concatenate(pps_list, axis=0)        # (total_samples, 8, 128*sample_seconds)
    labels_array = np.concatenate(labels_list, axis=0)  # (total_samples,)
    
    # Add Channel Dimensions
    eeg_array = np.expand_dims(eeg_array, axis=-1)  # (N, 32, 128*sample_seconds, 1)
    pps_array = np.expand_dims(pps_array, axis=-1)    # (N, 8, 128*sample_seconds, 1)
    
    # train/test split
    train_eeg, test_eeg, train_pps, test_pps, train_labels, test_labels = train_test_split(
        eeg_array, pps_array, labels_array, test_size=0.2, random_state=42, stratify=labels_array)
    return train_eeg, train_pps, train_labels, test_eeg, test_pps, test_labels

# 9. Custom callback: If validation accruals do not improve for 7 epochs, stop learning after restoring the highest weight
class RestoreBestWeightsOnDrop(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_accuracy', min_delta=0.001, patience=7, verbose=2):
        super(RestoreBestWeightsOnDrop, self).__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.best = -np.Inf
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        if current > self.best + self.min_delta:
            self.best = current
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"\nValidation {self.monitor} did not improve for {self.patience} epochs. Stopping training and restoring best weights.")
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True

# 10. Final Learning and Evaluation (10-fold Cross Validation + Final Model Learning)
def train_multimodal(use_bayesian=False, max_iterations=20, sample_seconds=SAMPLE_SECONDS):
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    for subject in subjects:
        print(f"\n==== {subject} 데이터 로드 ====")
        try:
            train_eeg, train_pps, train_labels, test_eeg, test_pps, test_labels = load_multimodal_data(subject, sample_seconds=sample_seconds)
            print(" EEG Input Shape:", train_eeg.shape)
            print(" PPS Input Shape:", train_pps.shape)
        except Exception as e:
            print(f"{subject} 데이터 로드 실패: {e}")
            continue
        
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        
        # 10-fold Cross Validation (for training data)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_val_scores = []
        fold = 1
        for train_index, val_index in skf.split(train_eeg, train_labels):
            print(f"\n---- {subject} Fold {fold} ----")
            fold_train_eeg, fold_valid_eeg = train_eeg[train_index], train_eeg[val_index]
            fold_train_pps, fold_valid_pps = train_pps[train_index], train_pps[val_index]
            fold_train_labels, fold_valid_labels = train_labels[train_index], train_labels[val_index]

            # Data balancing and augmentation (factor=2)
            fold_train_eeg, fold_train_pps, fold_train_labels = balance_data(fold_train_eeg, fold_train_pps, fold_train_labels)
            fold_train_eeg, fold_train_pps, fold_train_labels = augment_all_data(
                fold_train_eeg, fold_train_pps, fold_train_labels,
                noise_std_eeg=0.01, noise_std_pps=0.005, factor=2)
            
            print(f"Fold {fold} 학습 샘플: EEG {fold_train_eeg.shape}, PPS {fold_train_pps.shape}, 라벨 {fold_train_labels.shape}")
            
            # Calculate alpha values and optimize lambda
            alpha_values = compute_class_weights(fold_train_labels)
            print(f"Fold {fold} alpha values: {alpha_values}")
            train_data_fold = ((fold_train_eeg, fold_train_pps), fold_train_labels)
            val_data_fold   = ((fold_valid_eeg, fold_valid_pps), fold_valid_labels)
            
            if use_bayesian:
                best_lambdas = bayesian_opt_lambda(train_data_fold, val_data_fold, alpha_values, max_iterations=max_iterations)
            else:
                best_lambdas = grid_search_lambda(train_data_fold, val_data_fold, alpha_values)
            
            lambda1, lambda2, lambda3 = map(float, best_lambdas)
            print(f"Fold {fold} 최적 가중치: λ1={lambda1}, λ2={lambda2}, λ3={lambda3}")
            
            # Create and compile models (directly specify loss)
            model = build_combined_model(num_classes=4, sample_seconds=sample_seconds)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss={
                    "Inter_Classification": focal_loss(alpha=alpha_values),
                    "EEG_Classification": focal_loss(alpha=alpha_values),
                    "PPS_Classification": focal_loss(alpha=alpha_values),
                    "Weight_Softmax": lambda y_true, y_pred: tf.constant(0.0)
                },
                loss_weights={
                    "Inter_Classification": lambda1,
                    "EEG_Classification": lambda2,
                    "PPS_Classification": lambda3,
                    "Weight_Softmax": 0.0
                },
                metrics={
                    "Inter_Classification": "accuracy",
                    "EEG_Classification": "accuracy",
                    "PPS_Classification": "accuracy",
                    "Weight_Softmax": None
                }
            )
            model.summary()
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_Inter_Classification_accuracy',
                    mode='max',
                    patience=7,
                    restore_best_weights=True,
                    verbose=2
                )
            ]
            train_labels_dict = get_labels_dict(fold_train_labels)
            valid_labels_dict = get_labels_dict(fold_valid_labels)
            
            history = model.fit(
                [fold_train_eeg, fold_train_pps],
                train_labels_dict,
                validation_data=([fold_valid_eeg, fold_valid_pps], valid_labels_dict),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=2
            )
            
            # Save learning history plot (by Inter_Classification)
            fig, axs = plt.subplots(1,2, figsize=(12,5))
            axs[0].plot(history.history['Inter_Classification_loss'], label='train loss')
            axs[0].plot(history.history['val_Inter_Classification_loss'], label='val loss')
            axs[0].set_title(f'Fold {fold} Loss')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Loss')
            axs[0].legend()
            
            axs[1].plot(history.history['Inter_Classification_accuracy'], label='train acc')
            axs[1].plot(history.history['val_Inter_Classification_accuracy'], label='val acc')
            axs[1].set_title(f'Fold {fold} Accuracy')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Accuracy')
            axs[1].legend()
            plt.tight_layout()
            plot_filename = os.path.join(subject_save_path, f"{subject}_fold{fold}_training.png")
            plt.savefig(plot_filename)
            plt.close()
            
            # Verification Evaluation by Fold
            eval_results = model.evaluate([fold_valid_eeg, fold_valid_pps], valid_labels_dict, verbose=2, return_dict=True)
            fold_val_acc = eval_results.get("Inter_Classification_accuracy")
            if fold_val_acc is None:
                fold_val_acc = eval_results.get("Inter_Classification_sparse_categorical_accuracy")
            if fold_val_acc is None:
                print("Available metrics:", eval_results.keys())
                raise ValueError("Inter Classification accuracy metric not found in evaluation results")
            
            fold_val_scores.append(fold_val_acc)
            print(f"Fold {fold} 검증 정확도: {fold_val_acc}")

            fold += 1
        
        avg_val_acc = np.mean(fold_val_scores)
        print(f"Subject {subject} 10-fold 평균 검증 정확도: {avg_val_acc}")
        
        # Final model learning (using full training data)
        train_eeg_bal, train_pps_bal, train_labels_bal = balance_data(train_eeg, train_pps, train_labels)
        train_eeg_aug, train_pps_aug, train_labels_aug = augment_all_data(
            train_eeg_bal, train_pps_bal, train_labels_bal, noise_std_eeg=0.01, noise_std_pps=0.005, factor=2)
        
        alpha_values_full = compute_class_weights(train_labels_aug)
        train_eeg_final, valid_eeg_final, train_pps_final, valid_pps_final, train_labels_final, valid_labels_final = train_test_split(
            train_eeg_aug, train_pps_aug, train_labels_aug, test_size=0.2, stratify=train_labels_aug, random_state=42)
        
        full_train_data = ((train_eeg_final, train_pps_final), train_labels_final)
        full_val_data   = ((valid_eeg_final, valid_pps_final), valid_labels_final)
        
        if use_bayesian:
            best_lambdas_final = bayesian_opt_lambda(full_train_data, full_val_data, alpha_values_full, max_iterations=max_iterations)
        else:
            best_lambdas_final = grid_search_lambda(full_train_data, full_val_data, alpha_values_full)
        
        lambda1_f, lambda2_f, lambda3_f = map(float, best_lambdas_final)
        print(f"최종 모델 최적 가중치: λ1={lambda1_f}, λ2={lambda2_f}, λ3={lambda3_f}")
        
        final_model = build_combined_model(num_classes=4, sample_seconds=sample_seconds)
        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={
                "Inter_Classification": focal_loss(alpha=alpha_values_full),
                "EEG_Classification": focal_loss(alpha=alpha_values_full),
                "PPS_Classification": focal_loss(alpha=alpha_values_full),
                "Weight_Softmax": lambda y_true, y_pred: tf.constant(0.0)
            },
            loss_weights={
                "Inter_Classification": lambda1_f,
                "EEG_Classification": lambda2_f,
                "PPS_Classification": lambda3_f,
                "Weight_Softmax": 0.0
            },
            metrics={
                "Inter_Classification": "accuracy",
                "EEG_Classification": "accuracy",
                "PPS_Classification": "accuracy",
                "Weight_Softmax": None
            }
        )
        final_model.summary()
        
        callbacks_final = [
            tf.keras.callbacks.EarlyStopping(monitor='val_Inter_Classification_accuracy', mode='max', patience=7, restore_best_weights=True, verbose=1)
        ]
        
        train_labels_dict_full = get_labels_dict(train_labels_final)
        valid_labels_dict_full = get_labels_dict(valid_labels_final)
        
        history_final = final_model.fit(
            [train_eeg_final, train_pps_final],
            train_labels_dict_full,
            validation_data=([valid_eeg_final, valid_pps_final], valid_labels_dict_full),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_final,
            verbose=2
        )
        
        # Save Final Model Learning History plot
        fig, axs = plt.subplots(1,2, figsize=(12,5))
        axs[0].plot(history_final.history['Inter_Classification_loss'], label='train loss')
        axs[0].plot(history_final.history['val_Inter_Classification_loss'], label='val loss')
        axs[0].set_title('Final Model Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        
        axs[1].plot(history_final.history['Inter_Classification_accuracy'], label='train acc')
        axs[1].plot(history_final.history['val_Inter_Classification_accuracy'], label='val acc')
        axs[1].set_title('Final Model Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()
        plt.tight_layout()
        plot_filename_final = os.path.join(subject_save_path, f"{subject}_final_training.png")
        plt.savefig(plot_filename_final)
        plt.close()
        
        # Save Model Weight
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        final_model.save_weights(weight_path)
        print(f" 모델 가중치 저장: {weight_path}")
        
        # Evaluation of final model test data
        test_labels_dict = get_labels_dict(test_labels)
        predictions = final_model.predict([test_eeg, test_pps])
        inter_pred = predictions[0]
        predicted_labels = np.argmax(inter_pred, axis=-1)
        report = classification_report(test_labels, predicted_labels,
                                       target_names=["Excited", "Relaxed", "Stressed", "Bored"],
                                       labels=[0,1,2,3], zero_division=0)
        print(f"\n {subject} 테스트 리포트\n{report}")
        
        report_path = os.path.join(subject_save_path, f"{subject}_test_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f" 테스트 리포트 저장: {report_path}")

if __name__ == "__main__":
    train_multimodal(use_bayesian=False, max_iterations=20, sample_seconds=SAMPLE_SECONDS)
