# Eye_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model

def build_eye_model(num_classes: int,
                    seq_length: int = None,
                    feature_dim: int = 30,
                    backbone_trainable: bool = True) -> tf.keras.Model:
    """
    Builds a model that gracefully handles both
      - 시퀀스 길이 >= 2: Conv1D + MaxPooling 블록
      - 시퀀스 길이 == 1: MLP(Dense) 블록
    """
    inputs = layers.Input(shape=(seq_length, feature_dim), name="eye_input")
    x = layers.LayerNormalization(name="layer_norm")(inputs)

    # 시퀀스 길이가 2 이상일 때만 Conv1D+Pooling 사용
    if seq_length is None or seq_length >= 2:
        # Conv-Pool 블록 1
        x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

        # Conv-Pool 블록 2
        x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

        # 시퀀스 전체를 요약
        x = layers.GlobalAveragePooling1D(name="gap")(x)
    else:
        # 시퀀스 길이가 1일 때: 프레임 단위 MLP
        x = layers.Flatten(name="flatten")(x)                    # (batch, feature_dim)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.5, name="dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    model = Model(inputs, outputs, name="EyeFeatureModel")

    # ── backbone_trainable=False 이면 conv-pool 블록만 동결하고
    #    분류 헤드(output)만 학습 가능하도록 설정
    if not backbone_trainable:
        # 전체 레이어 동결
        for layer in model.layers:
            layer.trainable = False
        # 이름이 "output"인 분류기 레이어만 학습 가능하도록 되돌리기
        model.get_layer("output").trainable = True

    return model
