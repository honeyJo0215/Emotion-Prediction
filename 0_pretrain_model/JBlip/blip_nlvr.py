import tensorflow as tf
from tensorflow.keras import Model, layers
from blip import create_vit
from EMCSP_1D_CNN import EMCSP_EEG_1DCNN_Encoder
import numpy as np
import os
from urllib.parse import urlparse

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model, url_or_filename):
    """
    Loading a checkpoint into this TF model is not implemented.
    """
    raise NotImplementedError("load_checkpoint for TF is not implemented.")

class BLIP_NLVR(Model):
    def __init__(
        self,
        seq_encoder: Model = None,
        # EEG‐CSP-1DCNN 인코더 하이퍼파라미터
        fs: int = 200,
        window_len: int = 200,
        apply_smoothing: bool = False,
        n_components: int = 8,
        hidden_dim: int = 128,
        image_size=480,
        vit='base',
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
    ):
        super().__init__()
        # Visual encoder
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1
        )
        # 1D sequence encoder (EEG/R-PPG)
        #if seq_encoder is None:
        #    raise ValueError("seq_encoder must be provided for non-text modality")
        #self.seq_encoder = seq_encoder
        if seq_encoder is None:
            seq_encoder = EMCSP_EEG_1DCNN_Encoder(
                fs=fs,
                window_len=window_len,
                apply_smoothing=apply_smoothing,
                n_components=n_components,
                hidden_dim=hidden_dim
            )
        self.seq_encoder = seq_encoder


        # Projection heads
        self.vision_proj = layers.Dense(embed_dim)
        self.seq_proj    = layers.Dense(embed_dim)

        # Classification head: takes [img0_feat; img1_feat; seq_feat]
        self.cls_head = tf.keras.Sequential([
            layers.Dense(embed_dim, activation='relu'),
            layers.Dense(2)
        ])

    def call(self, image, sequence, targets=None, training=False):
        """
        Args:
            image: Tensor of shape [2*B, H, W, 3], two images per example stacked along batch
            sequence: Tensor of shape [B, L_seq, C], e.g. EEG or rPPG time series
            targets: Optional int Tensor [B], labels 0/1 for training
            training: Bool, whether in training mode
        Returns:
            If training: scalar loss; else: [B,2] logits
        """
        # 1) Encode images
        #    Output shape [2*B, num_patches, vision_width]
        img_embeds = self.visual_encoder(image, training=training)
        # Recover batch size B
        B = tf.shape(sequence)[0]
        # Split into the two views
        img0, img1 = tf.split(img_embeds, num_or_size_splits=[B, B], axis=0)
        # Global image features (CLS token at index 0)
        img0_feat = self.vision_proj(img0[:, 0, :])  # [B, embed_dim]
        img1_feat = self.vision_proj(img1[:, 0, :])  # [B, embed_dim]

        # 2) Encode sequence
        #    sequence: [B, L_seq, C]
        seq_states = self.seq_encoder(sequence, training=training)  # [B, L_seq, hidden_size]
        seq_feat   = self.seq_proj(seq_states[:, 0, :])             # [B, embed_dim]

        # 3) Fuse and classify
        fused = tf.concat([img0_feat, img1_feat, seq_feat], axis=1)  # [B, 3*embed_dim]
        logits = self.cls_head(fused)                                # [B,2]

        if training:
            # Compute classification loss
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    targets, logits, from_logits=True
                )
            )
            return loss
        else:
            return logits

def blip_nlvr(pretrained='', **kwargs):
    model = BLIP_NLVR(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:", msg)
    return model
