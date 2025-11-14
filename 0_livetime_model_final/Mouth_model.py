import tensorflow as tf
from tensorflow.keras import layers, Model

def build_mouth_model(num_classes: int,
                      seq_length: int = None,
                      feature_dim: int = 30,
                      backbone_trainable: bool = True) -> tf.keras.Model:
    
    inputs = layers.Input(shape=(seq_length, feature_dim), name="mouth_input")
    
    x = layers.LayerNormalization(name="layer_norm")(inputs)

    if seq_length is None or seq_length >= 2:
        # 1D CNN
        # Conv-Pool 블록 1
        x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name="conv1")(x)
        x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

        # Conv-Pool 블록 2
        x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu", name="conv2")(x)
        x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        
    else: # seq_length가 1인 경우
        # 단일 데이터 처리 (MLP)
        x = layers.Flatten(name="flatten")(x)
        
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.5, name="dropout")(x)

    # 최종 분류
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    model = Model(inputs, outputs, name="FaceFeatureModel")

    # ── backbone_trainable=False 이면 conv-pool 블록만 동결하고
    #    분류 헤드(output)만 학습 가능하도록 설정
    if not backbone_trainable:
        # 전체 레이어 동결
        for layer in model.layers:
            layer.trainable = False
        # 이름이 "output"인 분류기 레이어만 학습 가능하도록 되돌리기
        model.get_layer("output").trainable = True

    return model
    

if __name__ == '__main__':
    NUM_CLASSES = 6
    FEATURE_DIM = 35

    print("--- 1. 시계열 데이터(sequence)를 처리하는 모델 구조 ---")
    # 예: 10초 분량의 데이터를 한 번에 처리 (길이=10)
    seq_model = build_mouth_model(num_classes=NUM_CLASSES, seq_length=10, feature_dim=FEATURE_DIM)
    seq_model.summary()

    print("\n\n--- 2. 단일 세그먼트(single segment)를 처리하는 모델 구조 ---")
    # 예: 10초 평균 특징값 1개만 처리 (길이=1)
    single_model = build_mouth_model(num_classes=NUM_CLASSES, seq_length=1, feature_dim=FEATURE_DIM)
    single_model.summary()