import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
# EEG & rPPG 1D CNN 인코더
from your_1dcnn_module import OneDCNNEncoder

# 공통: 다중 생체 신호 임베딩 생성 함수
def build_multi_bio_embedding(bios, bio_encoders, bio_proj):
    feats = []
    for mod, enc in bio_encoders.items():
        feats.append(enc(bios[mod]))            # (B, hid)
    cat = tf.concat(feats, axis=-1)           # (B, total_bio_dim)
    return bio_proj(cat)                     # (B, emb_dim)

# 1) Pretrain: EEG↔rPPG 대비학습 및 Matching
class BLIP_PretrainTF(Model):
    def __init__(self, hid_dim=128, emb_dim=256):
        super().__init__()
        # 1D CNN 인코더
        self.bio_enc = {
            'eeg':  OneDCNNEncoder(in_channels=16, hidden_dim=hid_dim),
            'rppg': OneDCNNEncoder(in_channels=1,  hidden_dim=hid_dim)
        }
        total_dim = 2 * hid_dim
        # projection heads
        self.proj_eeg  = layers.Dense(emb_dim)
        self.proj_rppg = layers.Dense(emb_dim)
        # ITM head
        self.itm_head = layers.Dense(2)

    def call(self, bios, training=False):
        # 모달리티별 피처 추출
        f_eeg  = self.bio_enc['eeg'](bios['eeg'])    # (B, hid)
        f_rppg = self.bio_enc['rppg'](bios['rppg'])  # (B, hid)
        # 대비학습용 임베딩
        emb_eeg  = tf.nn.l2_normalize(self.proj_eeg(f_eeg), axis=-1)
        emb_rppg = tf.nn.l2_normalize(self.proj_rppg(f_rppg), axis=-1)
        # ITM 분류 입력: concat(eeg, rppg)
        itm_in = tf.concat([f_eeg, f_rppg], axis=-1)
        itm_out = self.itm_head(itm_in)             # (B,2)
        return emb_eeg, emb_rppg, itm_out

# 2) Retrieval: EEG↔rPPG 유사도 검색
class BLIP_RetrievalTF(Model):
    def __init__(self, hid_dim=128, emb_dim=256):
        super().__init__()
        self.bio_enc = {
            'eeg':  OneDCNNEncoder(16, hid_dim),
            'rppg': OneDCNNEncoder(1,  hid_dim)
        }
        self.proj = layers.Dense(emb_dim)

    def call(self, bios):
        f_eeg  = self.bio_enc['eeg'](bios['eeg'])
        f_rppg = self.bio_enc['rppg'](bios['rppg'])
        emb_eeg  = tf.nn.l2_normalize(self.proj(f_eeg), axis=-1)
        emb_rppg = tf.nn.l2_normalize(self.proj(f_rppg), axis=-1)
        return emb_eeg, emb_rppg

# 3) NLVR: 두 신호 쌍에 대한 이진 분류
class BLIP_NLVRTF(Model):
    def __init__(self, hid_dim=128):
        super().__init__()
        self.bio_enc = {
            'sig0': OneDCNNEncoder(16, hid_dim),
            'sig1': OneDCNNEncoder(16, hid_dim)
        }
        self.cls_head = tf.keras.Sequential([
            layers.Dense(hid_dim, activation='relu'),
            layers.Dense(2)
        ])

    def call(self, bios):
        f0 = self.bio_enc['sig0'](bios['sig0'])
        f1 = self.bio_enc['sig1'](bios['sig1'])
        cat = tf.concat([f0, f1], axis=-1)
        return self.cls_head(cat)

# 4) VQA: 시계열 질의응답 (multi-output regression/classification 예시)
class BLIP_VQATF(Model):
    def __init__(self, hid_dim=128, out_dim=10):
        super().__init__()
        self.bio_enc = {
            'eeg':  OneDCNNEncoder(16, hid_dim),
            'rppg': OneDCNNEncoder(1,  hid_dim)
        }
        self.head = layers.Dense(out_dim)

    def call(self, bios, question=None, training=False):
        # question 인자 무시, 두 signal 특징 결합 후 head 통과
        f_eeg  = self.bio_enc['eeg'](bios['eeg'])
        f_rppg = self.bio_enc['rppg'](bios['rppg'])
        cat = tf.concat([f_eeg, f_rppg], axis=-1)
        return self.head(cat)

# 5) Base: 단일/다중 신호 인코딩 반환
class BLIP_BaseTF(Model):
    def __init__(self, hid_dim=128):
        super().__init__()
        self.bio_enc = {
            'eeg':  OneDCNNEncoder(16, hid_dim),
            'rppg': OneDCNNEncoder(1,  hid_dim)
        }

    def call(self, bios, mode):
        assert mode in ['eeg','rppg','multimodal']
        if mode in ['eeg','rppg']:
            return self.bio_enc[mode](bios[mode])
        else:
            f_eeg  = self.bio_enc['eeg'](bios['eeg'])
            f_rppg = self.bio_enc['rppg'](bios['rppg'])
            return tf.concat([f_eeg, f_rppg], axis=-1)

# 6) Decoder: 시퀀스 생성 대신 감독 signal 예측용 head
class BLIP_DecoderTF(Model):
    def __init__(self, hid_dim=128, out_dim=1):
        super().__init__()
        self.bio_enc = OneDCNNEncoder(16, hid_dim)
        self.head = layers.Dense(out_dim)

    def call(self, bios, prompt=None):
        f = self.bio_enc(bios['eeg'])
        return self.head(f)

# 7) ITM: 두 신호 매칭 분류
class BLIP_ITMTF(Model):
    def __init__(self, hid_dim=128):
        super().__init__()
        self.bio_enc = {
            'eeg':  OneDCNNEncoder(16, hid_dim),
            'rppg': OneDCNNEncoder(1,  hid_dim)
        }
        self.itm_head = layers.Dense(2)

    def call(self, bios):
        f_eeg  = self.bio_enc['eeg'](bios['eeg'])
        f_rppg = self.bio_enc['rppg'](bios['rppg'])
        cat = tf.concat([f_eeg, f_rppg], axis=-1)
        return self.itm_head(cat)
