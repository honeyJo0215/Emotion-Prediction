#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEAP 데이터셋 (EEG, PPG) 기반 멀티모달 감정 분류 모델
- 각 피험자(s01 ~ s22)는 이미 전처리되어 저장된 데이터 파일을 사용합니다.
- EEG 데이터: (40, 32, 128) → 모델 입력: (40, 32, 128, 1)
- PPG 데이터: (40, 128) → 모델 입력: (40, 128, 1, 1)
- 라벨: (40,) (Negative: 0, Positive: 1, Neutral: 2)
- 데이터 파일은 각 피험자별로 한 파일에 40개의 1초 segment가 저장되어 있습니다.
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU를 숨겨 CPU만 사용
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"    # GPU 메모리 자동 증가 방지
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # TensorFlow 로그 최소화

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Conv1D, Dense, Flatten, Dropout, AveragePooling2D, AveragePooling1D, DepthwiseConv2D,
    LayerNormalization, MultiHeadAttention, Input, Lambda, Add, Concatenate, Softmax
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.optimize import minimize  # Bayesian Optimization
import itertools  # Grid Search

# =============================================================================
# GPU 메모리 제한 (필요 시)
# =============================================================================
# def limit_gpu_memory(memory_limit_mib=8000):
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             tf.config.experimental.set_virtual_device_configuration(
#                 gpus[0],
#                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
#             )
#             print(f"GPU memory limited to {memory_limit_mib} MiB.")
#         except RuntimeError as e:
#             print(e)
#     else:
#         print("No GPU available, using CPU.")

# limit_gpu_memory(8000)

# =============================================================================
# 하이퍼파라미터(논문과 일치) 및 경로 설정
# =============================================================================
BATCH_SIZE = 32       # 테이블 4에 맞게
EPOCHS = 150          # 테이블 4에 맞게
LEARNING_RATE = 1e-4  # 

# 데이터 경로 (필요에 따라 수정)
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG"
PPG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_PPS" #PPS는 밴드(&로우)패스필터 적용 안한 input.
#PPS2 폴더의 파일은 필터 적용한 input. 두개다 써보고 결과를 비교해보자.
LABELS_PATH  = "/home/bcml1/2025_EMOTION/DEAP_three_labels"
SAVE_PATH    = "/home/bcml1/sigenv/_3주차_new_EEG+PPS_CNN/m_result1"
os.makedirs(SAVE_PATH, exist_ok=True)

# =============================================================================
# 1. Dual-Stream Feature Extractor (EEG & PPG)
# =============================================================================
def create_dual_stream_feature_extractor():
    """
    EEG와 PPG를 각각 CNN으로 특징 추출
    - EEG Branch: 입력 (32, 128, 1)
    - PPG Branch: 입력 (128, 1, 1)
    """
    ## EEG Branch
    eeg_input = Input(shape=(32, 128, 1), name="EEG_Input")
    x = Conv2D(16, (4, 16), strides=(2, 8), padding='valid', activation='relu')(eeg_input)
    x = DepthwiseConv2D((6, 6), strides=(3, 3), padding='valid', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(32, (1, 1), strides=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    eeg_output = Dense(128, activation="relu")(x)
    
    ## PPG Branch
    ppg_input = Input(shape=(128, 1, 1), name="PPG_Input")
    y = Conv2D(16, (4, 1), strides=(2, 1), padding='valid', activation='relu')(ppg_input)
    # y = DepthwiseConv2D((6, 1), strides=(3, 1), padding='valid', activation='relu')(y)
    y = DepthwiseConv2D((6, 1), strides=3, padding='valid', activation='relu')(y)
    y = AveragePooling2D(pool_size=(2, 1), strides=(2, 1))(y)
    y = Conv2D(32, (1, 1), strides=(1, 1), activation='relu')(y)
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    ppg_output = Dense(128, activation="relu")(y)
    # ppg_1d = Lambda(lambda x: tf.squeeze(x, axis=-1))(ppg_input)  # shape: (128, 1)
    # y = Conv1D(16, kernel_size=4, strides=2, padding='valid', activation='relu')(ppg_1d)
    # y = Conv1D(16, kernel_size=6, strides=3, padding='valid', activation='relu')(y)
    # y = AveragePooling1D(pool_size=2, strides=2)(y)
    # y = Conv1D(32, kernel_size=1, strides=1, activation='relu')(y)
    # y = Flatten()(y)
    # y = Dropout(0.5)(y)
    # ppg_output = Dense(128, activation="relu")(y)
    
    model = Model(inputs=[eeg_input, ppg_input], outputs=[eeg_output, ppg_output],
                  name="DualStreamFeatureExtractor")
    return model

# =============================================================================
# 2. Inter-Modality Fusion Module (Cross-Modal Transformer)
# =============================================================================
def create_inter_modality_fusion(eeg_features, ppg_features, num_heads=4, d_model=128, dropout_rate=0.25):
    """
    EEG와 PPG 간의 상호관계를 학습하는 Cross‑Modal Transformer
    """
    # EEG → PPG
    eeg_query = Dense(d_model)(eeg_features)
    ppg_key_value = Dense(d_model)(ppg_features)
    eeg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_query)
    ppg_key_value = Lambda(lambda x: tf.expand_dims(x, axis=1))(ppg_key_value)

    cross_att_1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_EEG_to_PPG")(
        query=eeg_query, key=ppg_key_value, value=ppg_key_value)
    cross_att_1 = Dropout(dropout_rate)(cross_att_1)
    cross_att_1 = Add()([eeg_query, cross_att_1])
    cross_att_1 = LayerNormalization(epsilon=1e-6)(cross_att_1)
    cross_att_1 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_att_1)
    
    # PPG → EEG
    ppg_query = Dense(d_model)(ppg_features)
    ppg_query = Lambda(lambda x: tf.expand_dims(x, axis=1))(ppg_query)
    eeg_key_value_2 = Dense(d_model)(eeg_features)
    eeg_key_value_2 = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_key_value_2)

    cross_att_2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_PPG_to_EEG")(
        query=ppg_query, key=eeg_key_value_2, value=eeg_key_value_2)
    cross_att_2 = Dropout(dropout_rate)(cross_att_2)
    cross_att_2 = Add()([ppg_query, cross_att_2])
    cross_att_2 = LayerNormalization(epsilon=1e-6)(cross_att_2)
    cross_att_2 = Lambda(lambda x: tf.squeeze(x, axis=1))(cross_att_2)

    fused_features = Concatenate(axis=-1)([cross_att_1, cross_att_2])
    fused_features = Dense(d_model, activation="relu", name="Fused_Linear")(fused_features)
    
    # Self-Attention Fusion: 두 번의 self-attention 블록 적용
    fused_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(fused_features)
    for i in range(2):
        sa = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name=f"SelfAttentionFusion_{i+1}")(
            query=fused_expanded, key=fused_expanded, value=fused_expanded)
        sa = Dropout(dropout_rate)(sa)
        fused_expanded = Add()([fused_expanded, sa])
        fused_expanded = LayerNormalization(epsilon=1e-6)(fused_expanded)
    fused_features = Lambda(lambda x: tf.squeeze(x, axis=1))(fused_expanded)
    return fused_features

# =============================================================================
# 3. Intra-Modality Encoding Module
# =============================================================================
def create_intra_modality_encoding(eeg_features, ppg_features, num_heads=4, d_model=128, dropout_rate=0.25):
    """
    EEG와 PPG 각각의 고유 특성을 보존 및 강화하는 Intra‑Modal Encoding
    """
    # EEG Self-Attention
    eeg_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(eeg_features)
    eeg_self_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_EEG")(
        query=eeg_expanded, key=eeg_expanded, value=eeg_expanded)
    eeg_self_att = Dropout(dropout_rate)(eeg_self_att)
    eeg_self_att = Add()([eeg_expanded, eeg_self_att])
    eeg_self_att = LayerNormalization(epsilon=1e-6)(eeg_self_att)
    eeg_self_att = Lambda(lambda x: tf.squeeze(x, axis=1))(eeg_self_att)
    
    # PPG Self-Attention
    ppg_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(ppg_features)
    ppg_self_att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_PPG")(
        query=ppg_expanded, key=ppg_expanded, value=ppg_expanded)
    ppg_self_att = Dropout(dropout_rate)(ppg_self_att)
    ppg_self_att = Add()([ppg_expanded, ppg_self_att])
    ppg_self_att = LayerNormalization(epsilon=1e-6)(ppg_self_att)
    ppg_self_att = Lambda(lambda x: tf.squeeze(x, axis=1))(ppg_self_att)
    
    return eeg_self_att, ppg_self_att

# =============================================================================
# 4. 전체 모델 구성 (EEG, PPG 입력 → Dual‑Stream Extraction → Fusion → 분류)
# =============================================================================
def build_combined_model(num_classes=3):
    """
    EEG와 PPG 데이터를 받아서  
      - Dual‑Stream Feature Extraction  
      - Cross‑Modal Fusion  
      - Intra‑Modal Encoding  
    을 거쳐 최종 분류 결과를 출력하는 모델 구성
    """
    eeg_input = Input(shape=(32, 128, 1), name="EEG_Input")
    ppg_input = Input(shape=(128, 1, 1), name="PPG_Input")
        
    dual_extractor = create_dual_stream_feature_extractor()
    eeg_features, ppg_features = dual_extractor([eeg_input, ppg_input])
    
    fused_features = create_inter_modality_fusion(eeg_features, ppg_features)
    eeg_encoded, ppg_encoded = create_intra_modality_encoding(eeg_features, ppg_features)
    
    inter_classification = Dense(num_classes, activation="softmax", name="Inter_Classification")(fused_features)
    eeg_classification   = Dense(num_classes, activation="softmax", name="EEG_Classification")(eeg_encoded)
    ppg_classification   = Dense(num_classes, activation="softmax", name="PPG_Classification")(ppg_encoded)
    
    concat_features = Concatenate()([fused_features, eeg_encoded, ppg_encoded])
    weights_logits = Dense(units=3, activation=None, name="Weight_Logits")(concat_features)
    weights = Softmax(axis=-1, name="Weight_Softmax")(weights_logits)
    
    model = Model(inputs=[eeg_input, ppg_input],
                  outputs=[inter_classification, eeg_classification, ppg_classification, weights],
                  name="Multimodal_Emotion_Classifier")
    return model

# =============================================================================
# 5. Grid Search 및 Bayesian Optimization (loss와 loss_weights 적용)
# =============================================================================
def get_labels_dict(labels):
    # 모든 출력에 대해 동일한 레이블 (네 번째는 더미 값)
    return {
        "Inter_Classification": labels,
        "EEG_Classification": labels,
        "PPG_Classification": labels,
        "Weight_Softmax": np.zeros((labels.shape[0], 3))
    }

def grid_search_lambda(train_data, val_data, alpha_values):
    lambda_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_acc = 0.0
    best_lambdas = (0.5, 0.3, 0.2)  # 기본값
    
    (train_eeg, train_ppg), train_labels = train_data
    (valid_eeg, valid_ppg), valid_labels = val_data
    train_labels_dict = get_labels_dict(train_labels)
    valid_labels_dict = get_labels_dict(valid_labels)
    
    for l1, l2, l3 in itertools.product(lambda_grid, repeat=3):
        if abs(l1 + l2 + l3 - 1.0) > 0.05:  # 합이 1이 아닌 경우 제외
            continue

        model = build_combined_model(num_classes=3)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss={
                        "Inter_Classification"  : focal_loss(alpha=alpha_values),
                        "EEG_Classification": focal_loss(alpha=alpha_values),
                        "PPG_Classification": focal_loss(alpha=alpha_values),
                        "Weight_Softmax": lambda y_true, y_pred: tf.constant(0.0)
                    },
                    loss_weights={
                        "Inter_Classification": l1,
                        "EEG_Classification": l2,
                        "PPG_Classification": l3,
                        "Weight_Softmax": 0.0
                    },
                    metrics={
                        "Inter_Classification": "accuracy",
                        "EEG_Classification": "accuracy",
                        "PPG_Classification": "accuracy",
                        "Weight_Softmax": None  # 또는 빈 리스트 []
                    }
                    )
        
        model.fit([train_eeg, train_ppg], train_labels_dict,
                  validation_data=([valid_eeg, valid_ppg], valid_labels_dict),
                  epochs=3, batch_size=BATCH_SIZE, verbose=0)
        eval_results = model.evaluate([valid_eeg, valid_ppg], valid_labels_dict, verbose=0)
        val_acc = eval_results[1]  

        if val_acc > best_acc:
            best_acc = val_acc
            best_lambdas = (l1, l2, l3)

    print(f"✅ 최적 λ 값 (Grid Search): {best_lambdas}")
    return best_lambdas  # 가중치 반환

def bayesian_opt_lambda(train_data, val_data, alpha_values, max_iterations=15):
    (train_eeg, train_ppg), train_labels = train_data
    (valid_eeg, valid_ppg), valid_labels = val_data
    train_labels_dict = get_labels_dict(train_labels)
    valid_labels_dict = get_labels_dict(valid_labels)
    
    def loss_function(lambdas):
        l1, l2, l3 = lambdas
        if abs(l1 + l2 + l3 - 1.0) > 0.05:
            return 1.0  # 잘못된 조합은 높은 loss 반환

        model = build_combined_model(num_classes=3)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={
                "Inter_Classification": "sparse_categorical_crossentropy",
                "EEG_Classification": "sparse_categorical_crossentropy",
                "PPG_Classification": "sparse_categorical_crossentropy",
                "Weight_Softmax": lambda y_true, y_pred: tf.constant(0.0)
            },
            loss_weights={
                "Inter_Classification": np.float32(l1),
                "EEG_Classification": np.float32(l2),
                "PPG_Classification": np.float32(l3),
                "Weight_Softmax": np.float32(0.0)
            },
            # metrics=["accuracy"]
            metrics={
                        "Inter_Classification": "accuracy",
                        "EEG_Classification": "accuracy",
                        "PPG_Classification": "accuracy",
                        "Weight_Softmax": None  # 또는 빈 리스트 []
                    }
        )

        model.fit(
            [train_eeg, train_ppg],
            train_labels_dict,
            validation_data=([valid_eeg, valid_ppg], valid_labels_dict),
            epochs=3, batch_size=BATCH_SIZE, verbose=0
        )
        eval_results = model.evaluate([valid_eeg, valid_ppg], valid_labels_dict, verbose=0)
        val_acc = eval_results[1]
        return -val_acc  # 높은 정확도를 찾기 위해 음수로 반환

    result = minimize(
        loss_function, x0=[0.5, 0.3, 0.2],
        bounds=[(0.1, 0.9), (0.1, 0.9), (0.1, 0.9)],
        options={'maxiter': max_iterations}
    )
    best_lambdas = result.x
    print(f"✅ 최적 λ 값 (Bayesian Optimization): {best_lambdas}")
    return best_lambdas

# =============================================================================
# 6. 클래스 빈도 기반 alpha 및 Focal Loss
# =============================================================================
def compute_class_weights(labels, num_classes=3):
    class_counts = np.bincount(labels, minlength=num_classes)
    total = np.sum(class_counts)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = total / (num_classes * class_counts)
    class_weights /= np.max(class_weights)
    return class_weights.astype(np.float32)

def focal_loss(alpha, gamma=2.0):
    def loss(y_true, y_pred):
        # y_true: 정수 레이블, y_pred: softmax 출력 (shape: [batch, num_classes])
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(alpha))  # shape: [batch, num_classes]
        # tf.argmax(y_true_one_hot, axis=-1) -> shape: [batch]
        # tf.gather(alpha, ...) -> shape: [batch]
        # expand dims to [batch, 1] for broadcasting
        alpha_factor = tf.expand_dims(tf.gather(alpha, tf.argmax(y_true_one_hot, axis=-1)), axis=-1)
        loss_val = -alpha_factor * y_true_one_hot * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred + 1e-7)
        return tf.reduce_mean(loss_val)
    return loss

# =============================================================================
# 7. 데이터 증강: 노이즈 추가 또는 신호 반전 (flip) 적용
# 데이터 증강 함수 수정: 전체 데이터셋에 대해 증강을 적용하여 원본 데이터의 5배 크기로 만듭니다.
# EEG는 noise_std=0.01, PPG는 noise_std=0.005 (각각의 Hz 특성을 고려)
# =============================================================================
def augment_sample(eeg_sample, ppg_sample, noise_std_eeg=0.01, noise_std_ppg=0.005):
    """
    하나의 EEG/PPG 샘플에 대해 무작위로 가우시안 노이즈 추가 또는 신호 반전을 적용.
    """
    aug_type = np.random.choice([0, 1])
    if aug_type == 0:
        eeg_aug = eeg_sample + np.random.normal(0, noise_std_eeg, size=eeg_sample.shape)
        ppg_aug = ppg_sample + np.random.normal(0, noise_std_ppg, size=ppg_sample.shape)
    else:
        eeg_aug = -eeg_sample
        ppg_aug = -ppg_sample
    return eeg_aug, ppg_aug

def balance_data(train_eeg, train_ppg, train_labels):
    """
    원본 데이터셋에서 각 클래스의 수를 동일하게 맞추기 위해 oversampling (with replacement)을 수행합니다.
    """
    unique, counts = np.unique(train_labels, return_counts=True)
    max_count = np.max(counts)
    balanced_eeg = []
    balanced_ppg = []
    balanced_labels = []
    for cls in unique:
        indices = np.where(train_labels == cls)[0]
        if len(indices) < max_count:
            oversampled = np.random.choice(indices, size=max_count, replace=True)
        else:
            oversampled = indices
        balanced_eeg.append(train_eeg[oversampled])
        balanced_ppg.append(train_ppg[oversampled])
        balanced_labels.append(train_labels[oversampled])
    balanced_eeg = np.concatenate(balanced_eeg, axis=0)
    balanced_ppg = np.concatenate(balanced_ppg, axis=0)
    balanced_labels = np.concatenate(balanced_labels, axis=0)
    perm = np.random.permutation(balanced_labels.shape[0])
    return balanced_eeg[perm], balanced_ppg[perm], balanced_labels[perm]

def augment_all_data(train_eeg, train_ppg, train_labels, noise_std_eeg=0.01, noise_std_ppg=0.005, factor=5):
    """
    전체 데이터셋에 대해, 각 샘플마다 (factor-1)개의 증강 샘플을 생성하여
    원본 데이터와 함께 총 factor배 크기의 데이터를 만듭니다.
    """
    N = train_eeg.shape[0]
    aug_eeg = []
    aug_ppg = []
    aug_labels = []
    for i in range(N):
        for _ in range(factor - 1):
            e_aug, p_aug = augment_sample(train_eeg[i], train_ppg[i], noise_std_eeg, noise_std_ppg)
            aug_eeg.append(e_aug)
            aug_ppg.append(p_aug)
            aug_labels.append(train_labels[i])
    aug_eeg = np.array(aug_eeg)
    aug_ppg = np.array(aug_ppg)
    aug_labels = np.array(aug_labels)
    all_eeg = np.concatenate([train_eeg, aug_eeg], axis=0)
    all_ppg = np.concatenate([train_ppg, aug_ppg], axis=0)
    all_labels = np.concatenate([train_labels, aug_labels], axis=0)
    perm = np.random.permutation(all_labels.shape[0])
    return all_eeg[perm], all_ppg[perm], all_labels[perm]

# =============================================================================
# 8. 데이터 로드 함수
# =============================================================================
def load_multimodal_data(subject):
    import glob
    eeg_pattern = os.path.join(EEG_DATA_PATH, f"{subject}_eeg_signals_segment_*.npy")
    ppg_pattern = os.path.join(PPG_DATA_PATH, f"{subject}_ppg_signals_segment_*.npy")
    
    eeg_files = sorted(glob.glob(eeg_pattern))
    ppg_files = sorted(glob.glob(ppg_pattern))
    
    if len(eeg_files) == 0 or len(ppg_files) == 0:
        raise FileNotFoundError(f"{subject}에 해당하는 EEG 또는 PPG 파일을 찾을 수 없습니다.")
    if len(eeg_files) != len(ppg_files):
        print("경고: EEG와 PPG 파일 수가 일치하지 않습니다.")
    
    eeg_segments = []
    ppg_segments = []
    
    for eeg_file, ppg_file in zip(eeg_files, ppg_files):
        eeg_data = np.load(eeg_file)  # shape: (40, 32, 128)
        eeg_segments.append(eeg_data)
        ppg_data = np.load(ppg_file)  # shape: (40, 128)
        ppg_segments.append(ppg_data)
    
    eeg_array = np.concatenate(eeg_segments, axis=0)  # (세그먼트 수*40, 32, 128)
    ppg_array = np.concatenate(ppg_segments, axis=0)  # (세그먼트 수*40, 128)
    
    eeg_array = np.expand_dims(eeg_array, axis=-1)     # (N, 32, 128, 1)
    ppg_array = ppg_array.reshape(ppg_array.shape[0], ppg_array.shape[1], 1, 1)  # (N, 128, 1, 1)
    
    label_file = os.path.join(LABELS_PATH, f"{subject}_three_labels.npy")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"라벨 파일이 없습니다: {label_file}")
    labels = np.load(label_file)  # (40,)
    
    S = len(eeg_files)
    labels_repeated = np.tile(labels, S)  # (S*40,)
    
    train_eeg, test_eeg, train_ppg, test_ppg, train_labels, test_labels = train_test_split(
        eeg_array, ppg_array, labels_repeated, test_size=0.2, random_state=42, stratify=labels_repeated)
    
    return train_eeg, train_ppg, train_labels, test_eeg, test_ppg, test_labels

# =============================================================================
# 9. 학습 및 평가
# =============================================================================
def train_multimodal(use_bayesian=True, max_iterations=20):
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    for subject in subjects:
        print(f"\n==== {subject} 데이터 로드 ====")
        try:
            train_eeg, train_ppg, train_labels, test_eeg, test_ppg, test_labels = load_multimodal_data(subject)
        except Exception as e:
            print(f"{subject} 데이터 로드 실패: {e}")
            continue
        
        # Validation set 구성 (train set의 20%)
        train_eeg, valid_eeg, train_ppg, valid_ppg, train_labels, valid_labels = train_test_split(
            train_eeg, train_ppg, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
        
        # 데이터 증강 (소수 클래스 보완)
        # train_eeg, train_ppg, train_labels = augment_minority_data(train_eeg, train_ppg, train_labels, noise_std=0.01)
        train_eeg, train_ppg, train_labels = balance_data(train_eeg, train_ppg, train_labels)
        train_eeg, train_ppg, train_labels = augment_all_data(
            train_eeg, train_ppg, train_labels,
            noise_std_eeg=0.01, noise_std_ppg=0.005, factor=5)
        print(f"학습 샘플: EEG {train_eeg.shape}, PPG {train_ppg.shape}, 라벨 {train_labels.shape}")

        # alpha_values 먼저 계산
        alpha_values = compute_class_weights(train_labels)
        print(f"alpha values: {alpha_values}")
        
        # Grid Search or Bayesian Optimization 수행
        train_data = ((train_eeg, train_ppg), train_labels)
        val_data = ((valid_eeg, valid_ppg), valid_labels)
        
        if use_bayesian:
            best_lambdas = bayesian_opt_lambda(train_data, val_data, alpha_values, max_iterations=max_iterations)
        else:
            best_lambdas = grid_search_lambda(train_data, val_data, alpha_values)

        lambda1, lambda2, lambda3 = map(float, best_lambdas)
        print(f"🎯 최적 가중치 적용: λ1={lambda1}, λ2={lambda2}, λ3={lambda3}")

        # 모델 생성 및 학습 (이하 동일)
        base_model = build_combined_model(num_classes=3)
        model = base_model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={
                "Inter_Classification": focal_loss(alpha=alpha_values),
                "EEG_Classification": focal_loss(alpha=alpha_values),
                "PPG_Classification": focal_loss(alpha=alpha_values),
                "Weight_Softmax": lambda y_true, y_pred: tf.constant(0.0)
            },
            loss_weights={
                "Inter_Classification": lambda1,
                "EEG_Classification": lambda2,
                "PPG_Classification": lambda3,
                "Weight_Softmax": 0.0
            },
            # metrics=["accuracy"]
            metrics={
                        "Inter_Classification": "accuracy",
                        "EEG_Classification": "accuracy",
                        "PPG_Classification": "accuracy",
                        "Weight_Softmax": None  # 또는 빈 리스트 []
                    }
        )
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
            #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=2)
        ]
        
        train_labels_dict = get_labels_dict(train_labels)
        valid_labels_dict = get_labels_dict(valid_labels)
        
        history = model.fit(
            [train_eeg, train_ppg],
            train_labels_dict,
            validation_data=([valid_eeg, valid_ppg], valid_labels_dict),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2
        )

        # 모델 저장 및 평가 (이하 동일)
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        model.save_weights(weight_path)
        print(f"✅ 모델 가중치 저장: {weight_path}")

        predictions = model.predict([test_eeg, test_ppg])
        inter_pred = predictions[0]
        predicted_labels = np.argmax(inter_pred, axis=-1)
        report = classification_report(test_labels, predicted_labels,
                                       target_names=["Negative", "Positive", "Neutral"],
                                       labels=[0,1,2], zero_division=0)
        print(f"\n📊 {subject} 테스트 리포트\n{report}")

        report_path = os.path.join(subject_save_path, f"{subject}_test_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"✅ 테스트 리포트 저장: {report_path}")

if __name__ == "__main__":
    train_multimodal(use_bayesian=True, max_iterations=20)
