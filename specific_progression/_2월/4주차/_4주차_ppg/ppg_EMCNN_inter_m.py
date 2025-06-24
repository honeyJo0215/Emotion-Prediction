# 논문 그대로 구현한 코드
# Epochs: 300, Batch Size: 128, Optimizer: Adam (β1=0.9, β2=0.99)
# Loss Function: Cross-Entropy Loss
# Regularization: L2 Regularization (λ=0.001)
# Learning Rate: 0.001
# Inter-Subject Validation (7:3 비율로 데이터 분할)
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, DepthwiseConv1D, Dense, GlobalAveragePooling1D, concatenate, BatchNormalization, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# GPU 메모리 제한 (필요 시)
# =============================================================================
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

# === EMCNN 입력 데이터 형태 ===
input_shape = (5, 1280)  # (5 transformation branches, 1280 samples)
num_classes = 4  # 감정 클래스 (HVHA, HVLA, LVHA, LVLA)

# === MobileNet 기반 Feature Extraction CNN ===
def emcnn_branch(input_layer):
    """ EMCNN의 각 branch에 해당하는 CNN 블록 (MobileNet 기반) """
    x = Conv1D(8, kernel_size=7, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    
    # Depthwise Separable Convolution 적용
    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1, depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(16, kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1, depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(32, kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1, depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1, depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)

    # Global Average Pooling 적용
    x = GlobalAveragePooling1D()(x)
    return x

# === EMCNN 전체 모델 ===
def build_emcnn():
    inputs = Input(shape=(5, 1280))  # 5개의 변환 데이터를 입력으로 받음

    # ✅ Lambda Layer를 사용하여 KerasTensor 차원 확장
    identity_branch = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, 0, :], axis=-1))(inputs))  # (batch, 1280, 1)
    smooth_branch_s2 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, 1, :], axis=-1))(inputs))
    smooth_branch_s3 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, 2, :], axis=-1))(inputs))
    downsample_branch_d2 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, 3, :], axis=-1))(inputs))
    downsample_branch_d3 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, 4, :], axis=-1))(inputs))

    # === Branch Concatenation ===
    merged_features = concatenate([identity_branch, smooth_branch_s2, smooth_branch_s3, downsample_branch_d2, downsample_branch_d3])

    # === Fully Connected Layers (L2 Regularization 포함) ===
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(merged_features)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    output = Dense(num_classes, activation='softmax')(x)  # 감정 클래스 예측

    # === 모델 생성 ===
    model = Model(inputs=inputs, outputs=output)
    return model

# -------------------------------
# 학습 및 테스트 데이터 분할 (논문 설정 반영: 7:3)
# -------------------------------
def train_model(subject_id, X, y, num_classes=4, epochs=300, batch_size=128):
    print(f"Subject {subject_id} 학습 시작")

    # ✅ 데이터 분할 (논문 설정: 7:3 비율)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # ✅ 결과 저장 폴더 설정
    result_dir = f"EMCNN_Results/{subject_id}"
    os.makedirs(result_dir, exist_ok=True)

    model = build_emcnn()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=1)

    # ✅ 모델 저장
    model.save(os.path.join(result_dir, f"{subject_id}_model.keras"))

    # ✅ 학습 히스토리 저장
    np.save(os.path.join(result_dir, f"{subject_id}_history.npy"), history.history)

    # ✅ 평가 결과 저장
    y_pred = np.argmax(model.predict(X_test), axis=1)
    eval_path = os.path.join(result_dir, f"{subject_id}_evaluation.txt")
    with open(eval_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))

    # ✅ 혼동 행렬 이미지 저장
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{subject_id} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(result_dir, f"{subject_id}_confusion_matrix.png"))
    plt.close()

# -------------------------------
# 데이터 로드 및 학습 실행
# -------------------------------
def load_data(data_dir):
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    X, y = [], []

    for file_path in file_paths:
        data = np.load(file_path)
        label = int(file_path.split('_')[-1].split('.')[0])
        X.append(data)
        y.append(np.full((data.shape[0],), label))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

if __name__ == "__main__":
    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'
    X, y = load_data(data_dir)
    train_model('subject_all', X, y)
    
# # -------------------------------
# # 5-Fold Cross Validation 적용 학습
# # -------------------------------
# def train_model_for_subject(subject_id, X, y, num_classes=4, epochs=300, batch_size=32):
#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#     fold_no = 1

#     # 결과 저장 폴더 설정
#     result_dir = f"EMCNN_real_Results/{subject_id}"
#     os.makedirs(result_dir, exist_ok=True)

#     for train_index, test_index in kfold.split(X):
#         print(f"Subject {subject_id} - Fold {fold_no} 학습 시작")
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         model = build_emcnn()
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                       metrics=['accuracy'])

#         history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
#                             validation_data=(X_test, y_test), verbose=1)

#         # ✅ 모델 저장
#         model.save(os.path.join(result_dir, f"{subject_id}_fold{fold_no}_model.keras"))

#         # ✅ 학습 히스토리 저장
#         np.save(os.path.join(result_dir, f"{subject_id}_fold{fold_no}_history.npy"), history.history)

#         # ✅ 평가 결과 저장
#         y_pred = np.argmax(model.predict(X_test), axis=1)
#         eval_path = os.path.join(result_dir, f"{subject_id}_fold{fold_no}_evaluation.txt")
#         with open(eval_path, 'w') as f:
#             f.write(classification_report(y_test, y_pred))
#             f.write("\nConfusion Matrix:\n")
#             f.write(str(confusion_matrix(y_test, y_pred)))

#         # ✅ 혼동 행렬 이미지 저장
#         plt.figure(figsize=(6, 5))
#         sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
#         plt.title(f"{subject_id} Fold {fold_no} Confusion Matrix")
#         plt.xlabel("Predicted Label")
#         plt.ylabel("True Label")
#         plt.savefig(os.path.join(result_dir, f"{subject_id}_fold{fold_no}_confusion_matrix.png"))
#         plt.close()

#         fold_no += 1


# # -------------------------------
# # 데이터 로드 및 전체 서브젝트 학습 실행
# # -------------------------------
# def load_data(data_dir):
#     file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
#     X, y = [], []

#     for file_path in file_paths:
#         data = np.load(file_path)
#         label = int(file_path.split('_')[-1].split('.')[0])
#         X.append(data)
#         y.append(np.full((data.shape[0],), label))

#     X = np.concatenate(X, axis=0)
#     y = np.concatenate(y, axis=0)
#     return X, y


# if __name__ == "__main__":
#     data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'
#     X, y = load_data(data_dir)
#     train_model_for_subject('subject_all', X, y)
