# CNN 기반의 Dual-Stream Feature Extractor + Cross-Modal Transformer

import os
import re
import cv2  # downsample 시 사용
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, Dropout, AveragePooling2D, DepthwiseConv2D, LayerNormalization, 
    MultiHeadAttention, Reshape, Concatenate, GlobalAveragePooling2D, Input, TimeDistributed, 
    LSTM, Add, Dropout, Softmax
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# 📌 감정 라벨 매핑
EMOTION_MAPPING = {
    "Negative": 0,
    "Positive": 1,
    "Neutral": 2
}

# 📌 하이퍼파라미터 설정
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4

# 📌 데이터 경로
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap_labeled"
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
SAVE_PATH = "/home/bcml1/sigenv/_2주차_eye+eeg_CNN/result_e+ec"
os.makedirs(SAVE_PATH, exist_ok=True)
SUBJECTS = [f"s{str(i).zfill(2)}" for i in range(1, 23)]  # 피실험자 1~22명

# -------------------------
# 📌 Dual-Stream Feature Extractor
def create_dual_stream_feature_extractor():
    """
    EEG와 Eye Crop 데이터를 개별적으로 처리하는 Dual-Stream Feature Extractor
    """

    # 📌 EEG Stream (32채널, 1280 샘플 → (32, 1280, 1))
    eeg_input = Input(shape=(32, 1280, 1), name="EEG_Input")

    x = Conv2D(16, kernel_size=(3,3), activation="relu", padding="same")(eeg_input)  # Primary Convolution Layer
    x = DepthwiseConv2D(kernel_size=(3,3), activation="relu", padding="same")(x)  # Depthwise Convolution Layer
    x = AveragePooling2D(pool_size=(2,2))(x)  # Average Pooling Layer
    x = Conv2D(32, kernel_size=(3,3), activation="relu", padding="same")(x)  # Pointwise Convolution Layer
    x = Flatten()(x)  # Flatten Layer
    eeg_output = Dense(128, activation="relu")(x)  # Feature 벡터 변환

    # 📌 Eye Crop Stream (500, 8, 64, 3)
    eye_input = Input(shape=(500, 8, 64, 3), name="Eye_Input")

    # 📌 시간축(500 프레임) 별로 CNN 적용을 위해 TimeDistributed 사용
    eye_cnn = TimeDistributed(Conv2D(16, kernel_size=(1,16), activation="relu", padding="same"))(eye_input)
    eye_cnn = TimeDistributed(DepthwiseConv2D(kernel_size=(4,6), activation="relu", padding="same"))(eye_cnn)
    eye_cnn = TimeDistributed(AveragePooling2D(pool_size=(3,2)))(eye_cnn)
    eye_cnn = TimeDistributed(Conv2D(32, kernel_size=(1,1), activation="relu", padding="same"))(eye_cnn)
    eye_cnn = TimeDistributed(Flatten())(eye_cnn)  # 각 프레임(8,64,3)에서 특징 벡터 추출
    eye_cnn = TimeDistributed(Dense(64, activation="relu"))(eye_cnn)

    # 📌 LSTM을 적용하여 시간축(500) 프레임에 대한 요약
    eye_lstm = LSTM(128, return_sequences=False, dropout=0.3)(eye_cnn)  # 최종 128차원 특징 벡터
    eye_output = Dense(128, activation="relu")(eye_lstm)

    # 📌 최종 모델 생성
    model = Model(inputs=[eeg_input, eye_input], outputs=[eeg_output, eye_output], name="DualStreamFeatureExtractor")
    
    return model

# -------------------------
# 📌 Inter-Modality Fusion Module (Cross-Modal Transformer)
def create_inter_modality_fusion(eeg_features, eye_features, num_heads=4, d_model=128, dropout_rate=0.1):
    """
    EEG와 Eye Crop 간의 관계를 학습하는 Inter-Modality Fusion Module (Cross-Modal Transformer)
    
    Args:
        eeg_features: EEG에서 추출된 특징 벡터 (128차원)
        eye_features: Eye Crop에서 추출된 특징 벡터 (128차원)
        num_heads: Multi-Head Attention에서 사용할 헤드 수
        d_model: Feature dimension (128)
        dropout_rate: Dropout 비율

    Returns:
        최종 융합된 특징 벡터 (128차원)
    """

    # 1️⃣ **Cross-Modal Transformer: EEG → Eye Crop**
    eeg_query = Dense(d_model)(eeg_features)  # Query from EEG
    eye_key_value = Dense(d_model)(eye_features)  # Key, Value from Eye Crop

    cross_modal_attention_1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_EEG_to_Eye")(
        query=eeg_query, key=eye_key_value, value=eye_key_value
    )
    cross_modal_attention_1 = Dropout(dropout_rate)(cross_modal_attention_1)
    cross_modal_attention_1 = Add()([eeg_query, cross_modal_attention_1])
    cross_modal_attention_1 = LayerNormalization(epsilon=1e-6)(cross_modal_attention_1)

    # 2️⃣ **Cross-Modal Transformer: Eye Crop → EEG**
    eye_query = Dense(d_model)(eye_features)  # Query from Eye Crop
    eeg_key_value = Dense(d_model)(eeg_features)  # Key, Value from EEG

    cross_modal_attention_2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="CrossModal_Eye_to_EEG")(
        query=eye_query, key=eeg_key_value, value=eeg_key_value
    )
    cross_modal_attention_2 = Dropout(dropout_rate)(cross_modal_attention_2)
    cross_modal_attention_2 = Add()([eye_query, cross_modal_attention_2])
    cross_modal_attention_2 = LayerNormalization(epsilon=1e-6)(cross_modal_attention_2)

    # 3️⃣ **융합된 특징 벡터 생성**
    fused_features = tf.keras.layers.Concatenate(axis=-1)([cross_modal_attention_1, cross_modal_attention_2])  # 병합
    fused_features = Dense(d_model, activation="relu", name="Fused_Linear")(fused_features)

    # 4️⃣ **Self-Attention Transformer**
    self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttentionFusion")(
        query=fused_features, key=fused_features, value=fused_features
    )
    self_attention = Dropout(dropout_rate)(self_attention)
    self_attention = Add()([fused_features, self_attention])
    self_attention = LayerNormalization(epsilon=1e-6)(self_attention)

    return self_attention  # 최종 128차원 특징 벡터 출력

# Intra-Modality Encoding Module
def create_intra_modality_encoding(eeg_features, eye_features, num_heads=4, d_model=128, dropout_rate=0.1):
    """
    EEG와 Eye Crop의 고유한 특성을 유지하며 강화하는 Intra-Modality Encoding Module (Self-Attention Transformer)
    
    Args:
        eeg_features: EEG에서 추출된 특징 벡터 (128차원)
        eye_features: Eye Crop에서 추출된 특징 벡터 (128차원)
        num_heads: Multi-Head Attention에서 사용할 헤드 수
        d_model: Feature dimension (128)
        dropout_rate: Dropout 비율

    Returns:
        EEG & Eye Crop 각각의 Self-Attention을 적용한 개별 특성 벡터 (128차원)
    """

    # 📌 **Self-Attention Transformer: EEG Encoding**
    eeg_self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_EEG")(
        query=eeg_features, key=eeg_features, value=eeg_features
    )
    eeg_self_attention = Dropout(dropout_rate)(eeg_self_attention)
    eeg_self_attention = Add()([eeg_features, eeg_self_attention])
    eeg_self_attention = LayerNormalization(epsilon=1e-6)(eeg_self_attention)

    # 📌 **Self-Attention Transformer: Eye Crop Encoding**
    eye_self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name="SelfAttention_Eye")(
        query=eye_features, key=eye_features, value=eye_features
    )
    eye_self_attention = Dropout(dropout_rate)(eye_self_attention)
    eye_self_attention = Add()([eye_features, eye_self_attention])
    eye_self_attention = LayerNormalization(epsilon=1e-6)(eye_self_attention)

    return eeg_self_attention, eye_self_attention


def create_classifier(eeg_features, eye_features, num_classes=3):
    """
    최종 감정 분류기 (EEG + Eye Crop Feature 융합 후 Softmax 기반 분류)
    
    Args:
        inter_features: Inter-Modality Fusion을 거친 최종 특징 벡터 (128차원)
        eeg_features: Intra-Modality Encoding을 거친 EEG 특징 벡터 (128차원)
        eye_features: Intra-Modality Encoding을 거친 Eye Crop 특징 벡터 (128차원)
        num_classes: 감정 클래스 수 (기본값: 3)
    
    Returns:
        최종 모델 (Softmax 분류기 적용)
    """
    # 📌 **Inter-Modality Fusion을 내부에서 생성 (입력 필요 없음)**
    fused_features = create_inter_modality_fusion(eeg_features, eye_features)

    # 📌 **융합된 Inter-Modality Feature를 통한 감정 분류**
    inter_classification = Dense(num_classes, activation="softmax", name="Inter_Classification")(fused_features)

    # 📌 **EEG 단일 모달 Feature 분류**
    eeg_classification = Dense(num_classes, activation="softmax", name="EEG_Classification")(eeg_features)

    # 📌 **Eye Crop 단일 모달 Feature 분류**
    eye_classification = Dense(num_classes, activation="softmax", name="EyeCrop_Classification")(eye_features)

    # 📌 **가중치 학습을 위한 softmax 적용**
    # 3개의 예측 결과를 하나의 벡터로 만든 후 softmax를 적용하여 가중치로 변환
    

    # 📌 **가중치 학습을 위한 softmax 적용**
    concat_features = Concatenate()([fused_features, eeg_features, eye_features])  # ✅ Keras Concatenate 사용
    weights_logits = Dense(3)(concat_features)  
    weights = Softmax(axis=-1, name="Weight_Softmax")(weights_logits)  # (batch_size, 3)

    # 📌 **최종 모델 생성**
    model = Model(inputs=[eeg_features, eye_features], 
                  outputs=[inter_classification, eeg_classification, eye_classification, weights],
                  name="Multimodal_Emotion_Classifier")

    # 📌 **손실 함수 정의 (가중치 softmax 적용)**
    def multimodal_loss(y_true, y_preds):
        """
        L_total = W1 * L_inter + W2 * L_EEG + W3 * L_EyeCrop
        """
        inter_pred, eeg_pred, eye_pred, weight_logits = y_preds  # 모델의 4개 출력 (가중치 logits 포함)
        
        # 가중치 softmax 적용
        weight_softmax = tf.nn.softmax(weight_logits)  # (batch_size, 3)
        w1, w2, w3 = tf.split(weight_softmax, num_or_size_splits=3, axis=-1)  # 각각 분할
        
        # 개별 손실 계산
        loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
        loss_eeg = tf.keras.losses.sparse_categorical_crossentropy(y_true, eeg_pred)
        loss_eye = tf.keras.losses.sparse_categorical_crossentropy(y_true, eye_pred)

        # 각 손실에 동적으로 할당된 가중치 적용
        total_loss = (w1 * loss_inter) + (w2 * loss_eeg) + (w3 * loss_eye)

        return total_loss

    # 📌 **모델 컴파일**
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=multimodal_loss,
                  metrics=["accuracy"])

    return model

# -------------------------
# 📌 최종 Classifier
# def create_multimodal_emotion_classifier():
#     """
#     EEG + Eye Crop 기반 멀티모달 감정 분류 모델 생성.
#     """
#     # Feature Extractor
#     dual_stream_extractor = create_dual_stream_feature_extractor()
#     eeg_features, eye_features = dual_stream_extractor.output

#     # Inter-Modality Fusion
#     fused_features = create_inter_modality_fusion(eeg_features, eye_features)

#     # 최종 분류기
#     x = Dense(64, activation="relu")(fused_features)
#     x = Dropout(0.5)(x)
#     output = Dense(3, activation="softmax", name="Emotion_Output")(x)

#     return Model(inputs=dual_stream_extractor.input, outputs=output, name="MultimodalEmotionClassifier")


#데이터 로드 함수

# def find_subject_folder(base_path, subject):
#     """실제 파일 시스템에서 subject(s01, s02 ...)에 해당하는 폴더를 찾음."""
#     possible_folders = os.listdir(base_path)  # eye_crop 내 폴더 확인
#     for folder in possible_folders:
#         if folder.lower() == subject.lower():  # 대소문자 무시하고 비교
#             return os.path.join(base_path, folder)
#     return None  # 해당 폴더를 찾지 못한 경우

# # -------------------------
# # 📌 모델 학습 및 평가
# def train_multimodal_model():
#     model = create_multimodal_emotion_classifier()
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#                   loss="sparse_categorical_crossentropy",
#                   metrics=["accuracy"])

#     print(model.summary())

#     # 데이터 로드
#     eeg_data, eye_crop_data, labels = load_multimodal_data("s01")
#     X_train, X_test, y_train, y_test = train_test_split([eeg_data, eye_crop_data], labels, test_size=0.2, random_state=42)

#     # Dataset 생성
#     train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(8).shuffle(1000)
#     test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(8)

#     # 학습
#     model.fit(train_dataset, epochs=50, validation_data=test_dataset)

#     # 모델 저장
#     model.save(os.path.join(SAVE_PATH, "MultimodalEmotionModel.h5"))

# # -------------------------
# # 📌 실행
# if __name__ == "__main__":
#     train_multimodal_model()


# def build_full_model(num_classes=3):
#     """
#     EEG와 Eye Crop 데이터를 받아 Dual‑Stream Feature Extraction, 
#     Cross‑Modal Fusion, Intra‑Modal Encoding을 거쳐 3개의 분류 결과를 출력하는 전체 네트워크를 생성.
#     """
#     # 1. 입력 레이어 정의
#     eeg_input = tf.keras.layers.Input(shape=(32, 1280, 1), name="EEG_Input")
#     eye_input = tf.keras.layers.Input(shape=(500, 8, 64, 3), name="Eye_Input")
    
#     # 2. Dual‑Stream Feature Extraction (각각 128차원의 feature 벡터 산출)
#     dual_extractor = create_dual_stream_feature_extractor()
#     eeg_features, eye_features = dual_extractor([eeg_input, eye_input])
    
#     # 3. Inter‑Modality Fusion (Cross‑Modal Transformer)
#     fused_features = create_inter_modality_fusion(eeg_features, eye_features,
#                                                    num_heads=4, d_model=128, dropout_rate=0.1)
    
#     # 4. Intra‑Modality Encoding (Self‑Attention Transformer로 각 모달 특성 강화)
#     eeg_encoded, eye_encoded = create_intra_modality_encoding(eeg_features, eye_features,
#                                                               num_heads=4, d_model=128, dropout_rate=0.1)
    
#     # 5. 각 분류기 구성  
#     # # (A) 융합된 inter-modal feature를 통한 감정 분류  
#     # inter_output = tf.keras.layers.Dense(num_classes, activation="softmax", name="Inter_Classification")(fused_features)
#     # # (B) EEG 단일 모달 분류  
#     # eeg_output   = tf.keras.layers.Dense(num_classes, activation="softmax", name="EEG_Classification")(eeg_encoded)
#     # # (C) Eye Crop 단일 모달 분류  
#     # eye_output   = tf.keras.layers.Dense(num_classes, activation="softmax", name="EyeCrop_Classification")(eye_encoded)
    
#     # # 6. 최종 모델 생성
#     # model = tf.keras.models.Model(inputs=[eeg_input, eye_input],
#     #                               outputs=[inter_output, eeg_output, eye_output],
#     #                               name="Multimodal_Emotion_Classifier")
    
#     # # 7. 모델 컴파일: 각 분류기의 손실에 대해 가중합 (예, 0.5:0.25:0.25)
#     # loss_weights = {
#     #     "Inter_Classification": 0.5,
#     #     "EEG_Classification": 0.25,
#     #     "EyeCrop_Classification": 0.25
#     # }
    
#     # # 📌 **손실 함수 정의 (Cross-Entropy)**
#     # def multimodal_loss(y_true, y_pred):
#     #     """
#     #     L_total = λ1 * L_inter + λ2 * L_EEG + λ3 * L_EyeCrop
#     #     """
#     #     loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_classification)
#     #     loss_eeg = tf.keras.losses.sparse_categorical_crossentropy(y_true, eeg_classification)
#     #     loss_eye = tf.keras.losses.sparse_categorical_crossentropy(y_true, eye_classification)

#     #     return lambda1 * loss_inter + lambda2 * loss_eeg + lambda3 * loss_eye

    
#     # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#     #               loss={
#     #                   "Inter_Classification": "sparse_categorical_crossentropy",
#     #                   "EEG_Classification": "sparse_categorical_crossentropy",
#     #                   "EyeCrop_Classification": "sparse_categorical_crossentropy"
#     #               },
#     #               loss_weights=loss_weights,
#     #               metrics=["accuracy"])
    
#     model = create_classifier(eeg_encoded, eye_encoded, num_classes=3)
#     # inter_features, eeg_features, eye_features, num_classes=3
#     return model

def build_full_model(num_classes=3):
    # 1. 원본 입력 레이어 정의
    eeg_input = tf.keras.layers.Input(shape=(32, 1280, 1), name="EEG_Input")
    eye_input = tf.keras.layers.Input(shape=(500, 8, 64, 3), name="Eye_Input")
    
    # 2. Dual‑Stream Feature Extraction
    dual_extractor = create_dual_stream_feature_extractor()
    eeg_features, eye_features = dual_extractor([eeg_input, eye_input])
    
    # 3. Inter‑Modality Fusion
    fused_features = create_inter_modality_fusion(eeg_features, eye_features,
                                                   num_heads=4, d_model=128, dropout_rate=0.1)
    
    # 4. Intra‑Modality Encoding (개별 모달 특성 강화)
    eeg_encoded, eye_encoded = create_intra_modality_encoding(eeg_features, eye_features,
                                                              num_heads=4, d_model=128, dropout_rate=0.1)
    
    # 5. 분류기 층 구성
    inter_classification = Dense(num_classes, activation="softmax", name="Inter_Classification")(fused_features)
    eeg_classification   = Dense(num_classes, activation="softmax", name="EEG_Classification")(eeg_encoded)
    eye_classification   = Dense(num_classes, activation="softmax", name="EyeCrop_Classification")(eye_encoded)
    
    # 6. 가중치 학습 branch (세 분류 결과에 대한 가중치를 동적으로 학습)
    concat_features = Concatenate()([fused_features, eeg_encoded, eye_encoded])
    weights_logits = Dense(3)(concat_features)
    # Softmax에 axis=-1를 명시적으로 지정 (이렇게 하면 출력 텐서의 마지막 차원에 대해 softmax가 적용됩니다)
    weights = Softmax(axis=-1, name="Weight_Softmax")(weights_logits)
    
    # 7. 최종 모델 생성 (입력은 원본 입력 텐서 사용)
    model = Model(inputs=[eeg_input, eye_input],
                  outputs=[inter_classification, eeg_classification, eye_classification, weights],
                  name="Multimodal_Emotion_Classifier")
    
    # 8. 손실 함수 (예시)
    def multimodal_loss(y_true, y_preds):
        # y_preds는 4개 출력: inter, eeg, eye, weight_logits
        inter_pred, eeg_pred, eye_pred, weight_logits = y_preds  
        # 가중치 softmax 적용
        weight_softmax = tf.nn.softmax(weight_logits)  # (batch_size, 3)
        w1, w2, w3 = tf.split(weight_softmax, num_or_size_splits=3, axis=-1)
        loss_inter = tf.keras.losses.sparse_categorical_crossentropy(y_true, inter_pred)
        loss_eeg   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eeg_pred)
        loss_eye   = tf.keras.losses.sparse_categorical_crossentropy(y_true, eye_pred)
        total_loss = (w1 * loss_inter) + (w2 * loss_eeg) + (w3 * loss_eye)
        return total_loss

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=multimodal_loss,
                  metrics=["accuracy"])
    
    return model

# # ✅ **입력 데이터 크기 줄이기 (다운샘플링)**
def downsample_eye_frame(frame):
    """Eye Crop 이미지 다운샘플링 (64x32 → 32x16)"""
    return cv2.resize(frame, (32,8), interpolation=cv2.INTER_AREA)  # 해상도 절반 감소

# # ✅ **Eye Crop 데이터 로드 시 다운샘플링 적용**
def reshape_eye_frame(data):
    """
    (N, 32, 64, 3) 형태의 eye frame 데이터를 (32, 64, 3)으로 변환 후 다운샘플링 적용.
    - N이 2 이상이면 평균을 내서 병합.
    - N이 1이면 그대로 사용.
    """
    if len(data.shape) == 4 and data.shape[0] > 0:  
        reshaped_data = np.mean(data, axis=0)  # 모든 요소를 평균 내어 병합 (32, 64, 3)
        return downsample_eye_frame(reshaped_data)  # 다운샘플링 적용
    elif len(data.shape) == 3 and data.shape == (32, 64, 3):  
        return downsample_eye_frame(data)  # 다운샘플링 적용
    else:
        raise ValueError(f"Unexpected shape: {data.shape}, expected (N, 32, 64, 3) or (32, 64, 3)")


# # 입력받는 EEG data의 shape는 (32, 1280)이고, Eye Crop 의 shape은 (2, 32, 64, 3) -> (500, 32, 64, 6)이야.
def load_multimodal_data(subject):
    eeg_data, eye_data, labels = [], [], []

    # 파일 이름에 포함된 Sample 및 Segment 번호를 추출하는 정규표현식
    eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+).npy")

    # Sample 번호를 1부터 40까지 사용 (즉, Sample 01 ~ Sample 40)
    for sample_index in range(1, 2):
        sample_number = f"{sample_index:02d}"  
        print(f"\n🟢 Processing {subject} - Sample {sample_number}")

        eeg_files = [f for f in os.listdir(EEG_DATA_PATH)
                     if eeg_pattern.match(f) and f"sample_{sample_number}" in f]
        if not eeg_files:
            print(f"🚨 No EEG file found for {subject} - Sample {sample_number}")
            continue

        for file_name in eeg_files:
            match = eeg_pattern.match(file_name)
            if not match:
                continue

            # Segment 번호는 이제 001, 002, ... 로 되어 있으므로, 첫 Segment("001")가 1이 됨.
            segment_index = int(match.group(2))  # 예: "001" -> 1
            emotion_label = match.group(3)

            eeg_file_path = os.path.join(EEG_DATA_PATH, file_name)
            eeg_segment = np.load(eeg_file_path)

            eye_subject_path = os.path.join(EYE_CROP_PATH, subject)
            if not os.path.exists(eye_subject_path):
                print(f"🚨 Subject folder not found: {eye_subject_path}")
                continue

            # Trial 번호는 이제 sample_index 그대로 사용 (즉, Sample 01이면 Trial 01)
            trial_number = sample_index

            # 기존에는 non-overlap 방식으로 윈도우를 선택했다면,
            # 이제 sliding window (window_length=500, stride=50)를 적용함.
            # EEG 파일의 segment 번호에 따라 예상 시작 프레임은:
            expected_start = (segment_index - 1) * 500  # 예: segment "001" -> 0, "002" -> 500, ...
            # frame_indices: 해당 trial에서 존재하는 프레임 번호 목록
            frame_indices = set()
            file_mapping = {}

            for f in os.listdir(eye_subject_path):
                try:
                    if not f.startswith(subject) or not f.endswith(".npy"):
                        continue

                    match_frame = re.search(r"trial(\d+).avi_frame(\d+)", f)
                    if not match_frame:
                        print(f"⚠ Skipping invalid file name: {f} (No trial/frame pattern found)")
                        continue

                    file_trial_number = int(match_frame.group(1))
                    frame_number = int(match_frame.group(2))

                    if file_trial_number == trial_number:
                        frame_indices.add(frame_number)
                        file_mapping[frame_number] = os.path.join(eye_subject_path, f)

                except ValueError as e:
                    print(f"🚨 Error processing file {f}: {e}")
                    continue

            frame_indices = sorted(frame_indices)
            print(f"  🔍 Available frame indices: {frame_indices[:10]} ... {frame_indices[-10:]}")

            # --- 슬라이딩 윈도우 적용 ---
            window_length = 500
            stride = 50
            candidate_windows = []
            # candidate windows: stride 간격으로 500개의 프레임 목록 생성
            for i in range(0, len(frame_indices) - window_length + 1, stride):
                window = frame_indices[i:i+window_length]
                candidate_windows.append(window)
            if len(candidate_windows) == 0:
                print(f"⚠ Warning: Not enough frames for sliding window for segment {segment_index:03d}. Skipping Eye Crop.")
                eye_data.append(None)  # Eye Crop 데이터 없음
                eeg_data.append(eeg_segment)  # EEG 데이터는 추가
                labels.append(EMOTION_MAPPING[emotion_label])
                continue

            # 선택: candidate window 중 예상 시작(expected_start)과 첫 프레임 차이가 가장 작은 윈도우 선택
            selected_frames = min(candidate_windows, key=lambda w: abs(w[0] - expected_start))
            # --- 끝 ---

            # 만약 선택된 윈도우의 길이가 500이 아니라면 (드물게 발생할 경우) 복제하여 채움
            if len(selected_frames) < 500:
                print(f"⚠ Warning: Found only {len(selected_frames)} frames in selected window for segment {segment_index:03d}")
                while len(selected_frames) < 500:
                    selected_frames.append(selected_frames[-1])
                    print("프레임 복제됨")

            # eye_frame_files: 선택된 프레임 번호에 해당하는 파일 경로 목록 생성
            eye_frame_files = []
            for frame in selected_frames:
                if frame in file_mapping:
                    eye_frame_files.append(file_mapping[frame])
                if len(eye_frame_files) == 500:
                    break

            eye_frame_stack = []
            for f in eye_frame_files:
                frame_data = np.load(f)
                frame_data = reshape_eye_frame(frame_data)  # 반드시 미리 정의되어 있어야 함.
                # 만약 frame_data의 shape가 (32,64,3)라면, padding 적용하여 (64,64,3)로 변경
                if frame_data.shape[-2] == 32:
                    pad_width = [(0, 0)] * frame_data.ndim
                    pad_width[-2] = (16, 16)
                    frame_data = np.pad(frame_data, pad_width, mode='constant', constant_values=0)
                eye_frame_stack.append(frame_data)

            if len(eye_frame_stack) == 500:
                eye_data.append(np.stack(eye_frame_stack, axis=0))
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
            else:
                print(f"⚠ Warning: Found only {len(eye_frame_stack)} matching frames for segment {segment_index:03d}")

    print(f"✅ Total EEG Samples Loaded: {len(eeg_data)}")
    print(f"✅ Total Eye Crop Samples Loaded: {len([e for e in eye_data if e is not None])}")
    print(f"✅ Labels Loaded: {len(labels)}")

    return np.array(eeg_data), np.array([e if e is not None else np.zeros((500, 8, 64, 3)) for e in eye_data]), np.array(labels)

# 🟢 **학습 및 평가**
def train_multimodal():
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    for subject in subjects:
        print(f"Training subject: {subject}")
        
        # 데이터 로드
        eeg_data, eye_data, labels = load_multimodal_data(subject)

        # 샘플 단위로 Train/Valid/Test 데이터 나누기
        unique_samples = np.arange(len(eeg_data))  
        train_samples, test_samples = train_test_split(unique_samples, test_size=0.2, random_state=42)
        train_samples, valid_samples = train_test_split(train_samples, test_size=0.2, random_state=42)

        # 샘플 인덱스 기반으로 데이터 분할
        train_eeg, train_eye, train_labels = eeg_data[train_samples], eye_data[train_samples], labels[train_samples]
        valid_eeg, valid_eye, valid_labels = eeg_data[valid_samples], eye_data[valid_samples], labels[valid_samples]
        test_eeg, test_eye, test_labels = eeg_data[test_samples], eye_data[test_samples], labels[test_samples]

        # 모델이 학습 도중 OOM으로 종료될 경우 체크포인트를 저장하고 재시작하면 메모리 문제를 해결가능
        # 🚀 **각 subject 별 체크포인트 저장 경로 설정**
        checkpoint_dir = f"/home/bcml1/sigenv/_2주차_eye+eeg_CNN/checkpoint/cross_{subject}"
        checkpoint_path = os.path.join(checkpoint_dir, "cp.weights.h5")
        os.makedirs(checkpoint_dir, exist_ok=True)  # 디렉토리 없으면 생성

        # 체크포인트 콜백 설정 (자동 저장)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        )
        
        # 멀티모달 모델 구축
        model = build_full_model(num_classes=3)
        print(model.summary())

        # 🚀 **기존 체크포인트 로드 (있다면)**
        # if os.path.exists(checkpoint_path + ".index"):
        #     print(f"✅ Checkpoint found for {subject}, loading model...")
        #     model.load_weights(checkpoint_path)

        # 라벨 차원 확장
        train_labels = np.expand_dims(train_labels, axis=-1)
        valid_labels = np.expand_dims(valid_labels, axis=-1)

        # 🚀 **학습 파라미터 설정**
        start_epoch = 0
        max_epochs = 50
        batch_size = 2
        max_retries = 3  # 한 에포크당 최대 재시도 횟수

        # 에포크별 학습
        for epoch in range(start_epoch, max_epochs):
            retries = 0
            while retries < max_retries:
                try:
                    print(f"\n🚀 Training {subject} - Epoch {epoch+1}/{max_epochs} (Retry: {retries})...")
                    model.fit(
                        [train_eeg, train_eye], train_labels,
                        validation_data=([valid_eeg, valid_eye], valid_labels),
                        epochs=1, batch_size=batch_size
                        #callbacks=[checkpoint_callback]
                    )
                    # 에포크가 정상적으로 완료되면 while 루프 탈출
                    break  
                except tf.errors.ResourceExhaustedError:
                    print(f"⚠️ OOM 발생! 체크포인트 저장 후 GPU 메모리 정리 & 재시작 (Retry: {retries+1})...")
                    # 체크포인트 저장 시 OOM이 발생할 경우 예외 처리
                    try:
                        model.save_weights(checkpoint_path)
                    except tf.errors.ResourceExhaustedError:
                        print("⚠️ 체크포인트 저장 중 OOM 발생 - 저장 건너뜀.")
                    
                    # GPU 메모리 정리
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    # 모델 재생성 및 체크포인트 로드 (있다면)
                    model = build_full_model(eeg_input_shape=train_eeg.shape[1:])
                  
                    retries += 1
                    # 재시도 전에 잠시 휴식 (옵션)
                    tf.keras.backend.sleep(1)
            else:
                # 최대 재시도 횟수를 초과하면 에포크 종료 및 다음 subject로 넘어감.
                print(f"❌ 에포크 {epoch+1}에서 최대 재시도 횟수를 초과하였습니다. 다음 subject로 넘어갑니다.")
                break  # 또는 continue를 사용하여 다음 subject로 넘어갈 수 있음.

        # 🚀 **최종 모델 저장**
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        model.save_weights(weight_path)
        print(f"✅ 모델 가중치 저장됨: {weight_path}")

        # 🚀 **테스트 평가**
        predictions = model.predict([test_eeg, test_eye])
        predicted_labels = np.argmax(predictions, axis=-1)
        test_report = classification_report(
            test_labels, predicted_labels,
            target_names=["Negative", "Positive", "Neutral"],
            labels=[0, 1, 2],
            zero_division=0
        )
        print(f"\n📊 Test Report for {subject}")
        print(test_report)

        report_path = os.path.join(subject_save_path, f"{subject}_test_report.txt")
        with open(report_path, "w") as f:
            f.write(test_report)
        print(f"✅ 테스트 리포트 저장됨: {report_path}")
        
if __name__ == "__main__":
    train_multimodal()


# # ======================================================
# # ★ Intra‑Subject Cross‑Validation: subject별 데이터 로드 → 학습/테스트 분할 → 모델 학습 및 평가
# # ======================================================
# if __name__ == "__main__":
#     for subject in SUBJECTS:
#         print("="*50)
#         print(f"🔹 Processing Subject: {subject}")
        
#         # 1. 해당 subject의 데이터 로드
#         eeg_data, eye_data, labels = load_multimodal_data(subject)
        
#         if len(eeg_data) == 0:
#             print(f"🚨 Subject {subject}: 데이터가 없습니다. 다음 subject로 넘어갑니다.")
#             continue
        
#         print(f"✅ {subject} - Total Samples: {len(eeg_data)}")
        
#         # 2. 80:20 비율로 학습/테스트 데이터 분할 (레이블 불균형을 고려하여 stratify 사용)
#         X_train_eeg, X_test_eeg, X_train_eye, X_test_eye, y_train, y_test = train_test_split(
#             eeg_data, eye_data, labels, test_size=0.2, random_state=42, stratify=labels)
        
#         print(f"학습 샘플: {len(X_train_eeg)} | 테스트 샘플: {len(X_test_eeg)}")
        
#         # 3. subject별 모델 생성
#         model = build_full_model(num_classes=3)
#         model.summary()
        
#         # 4. 모델 학습
#         history = model.fit([X_train_eeg, X_train_eye],
#                             {"Inter_Classification": y_train,
#                              "EEG_Classification": y_train,
#                              "EyeCrop_Classification": y_train},
#                             validation_data=([X_test_eeg, X_test_eye],
#                                              {"Inter_Classification": y_test,
#                                               "EEG_Classification": y_test,
#                                               "EyeCrop_Classification": y_test}),
#                             epochs=EPOCHS,
#                             batch_size=BATCH_SIZE,
#                             verbose=1)
        
#         # 5. 테스트 데이터에 대한 평가
#         print(f"\n🔸 Evaluating Subject {subject}...")
#         eval_results = model.evaluate([X_test_eeg, X_test_eye],
#                                       {"Inter_Classification": y_test,
#                                        "EEG_Classification": y_test,
#                                        "EyeCrop_Classification": y_test},
#                                       batch_size=BATCH_SIZE, verbose=1)
#         print("Evaluation results:")
#         for name, value in zip(model.metrics_names, eval_results):
#             print(f"  {name}: {value:.4f}")
        
#         # 6. 예측 및 Classification Report (융합 분류 결과 기준)
#         predictions = model.predict([X_test_eeg, X_test_eye], batch_size=BATCH_SIZE)
#         # predictions의 첫 번째 출력(Inter_Classification)을 기준으로 최종 예측 산출
#         inter_preds = np.argmax(predictions[0], axis=1)
#         print(f"\nSubject {subject} Classification Report (Inter-Modal Output):")
#         print(classification_report(y_test, inter_preds, target_names=list(EMOTION_MAPPING.keys())))
        
#         # 7. 학습 완료된 subject별 모델 저장
#         subject_save_path = os.path.join(SAVE_PATH, subject)
#         os.makedirs(subject_save_path, exist_ok=True)
#         model_save_file = os.path.join(subject_save_path, "multimodal_emotion_classifier.h5")
#         model.save(model_save_file)
#         print(f"✅ Model for subject {subject} saved at {model_save_file}\n")