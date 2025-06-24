# PPG 시계열 3채널, HRV feature, 그리고 CWT 이미지까지 총 5개의 브랜치를 사용해서 feature-level fusion으로 4가지 감정 분류를 수행하는 구조로 모델을 설계할게. 각각의 modality에 최적화된 sub-network 구조를 적용하고, 최종적으로는 모든 branch의 feature를 concat해서 classification하도록 구성

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers

def build_ppg_cnn_lstm_model(
    seq_len=128,
    cwt_shape=(32,4,1),
    num_hrv_stats=4,        # 예: SDNN, RMSSD, LF/HF, etc.
    num_classes=4,
    l2_reg=1e-4
):
    # ── 1D PPG CNN+LSTM 브랜치 ─────────────────────────────────────────────
    def make_ppg_branch(name):
        inp = layers.Input((seq_len,1), name=name)
        x = layers.Conv1D(32, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(inp)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.MaxPooling1D(2)(x)
        # LSTM으로 시계열 의존성 학습
        x = layers.LSTM(64, kernel_regularizer=regularizers.l2(l2_reg))(x)
        return inp, x

    inp_o, feat_o = make_ppg_branch("ppg_orig")
    inp_s, feat_s = make_ppg_branch("ppg_smooth")
    inp_d, feat_d = make_ppg_branch("ppg_down")

    # ── HRV 통계 특징 MLP 브랜치 ────────────────────────────────────────────
    inp_h = layers.Input((seq_len,), name="ppg_hrv_series")
    # 추가로 SDNN, RMSSD 같은 요약 통계가 있다면 concat 하시고
    # 이하 예시는 시계열 HRV 벡터 하나만 사용하는 경우
    h = layers.BatchNormalization()(inp_h)
    h = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(h)
    h = layers.Dropout(0.3)(h)
    feat_h = layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(h)

    # ── CWT 2D-CNN 브랜치 ─────────────────────────────────────────────────
    inp_c = layers.Input(cwt_shape, name="ppg_cwt")
    z = layers.Conv2D(16, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(inp_c)
    z = layers.MaxPooling2D((2,2))(z)
    z = layers.Conv2D(32, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(z)
    z = layers.MaxPooling2D((2,2))(z)
    z = layers.GlobalAveragePooling2D()(z)
    feat_c = layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(z)

    # ── 모든 브랜치 합치고 분류기 ────────────────────────────────────────────
    merged = layers.concatenate([feat_o, feat_s, feat_d, feat_h, feat_c], name="merged_feats")
    x = layers.BatchNormalization()(merged)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax', name="emotion")(x)

    model = models.Model(
        inputs=[inp_o, inp_s, inp_d, inp_h, inp_c],
        outputs=out,
        name="PPG_CNN_LSTM_Emotion"
    )

    # 컴파일
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── 콜백 설정 예시 ─────────────────────────────────────────────────────────
es = callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)
rlr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
)

# ── 사용 예시 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 데이터 로드 (Xo, Xs, Xd, Xh, Xc), y 준비 후...
    model = build_ppg_cnn_lstm_model()
    model.summary()

    # train_ds, val_ds 준비 (tf.data.Dataset)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=200,
        callbacks=[es, rlr],
        verbose=2
    )
