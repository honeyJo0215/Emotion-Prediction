import tensorflow as tf
from tensorflow.keras import Model, layers
from blip import create_vit
from EMCSP_1D_CNN import EMCSP_EEG_1DCNN_Encoder
from typing import Tuple, Optional
import os

# Placeholder for gather operations in distributed setup
@tf.function
def concat_all_gather(tensor):
    return tensor

# Simple checkpoint loader for TensorFlow models
def load_checkpoint(model: Model, checkpoint_path: str) -> Tuple[Model, Optional[tf.train.Checkpoint]]:
    """
    Load model weights from a TensorFlow checkpoint directory or file.
    Returns the model and the checkpoint object (or None if not found).
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' does not exist.")
    # Create checkpoint and restore
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(checkpoint_path)
    # Optionally assert restoration
    status.assert_existing_objects_matched()
    return model, ckpt

class BLIP_ITM(Model):
    """
    BLIP ITM adapted for 1D signals (EEG/rPPG) instead of text.

    Args:
        seq_encoder: 1D CNN encoder for the signal
        image_size: input image size
        vit: vision transformer variant ('base' or 'large')
        embed_dim: projection dimension
    """
    def __init__(
        #self,
        #seq_encoder: Model = Sequence1DEncoder(in_channels=1, hidden_size=768, num_layers=4),
        self,
        seq_encoder: Model = None,
        # EEG‐CSP-1DCNN 인코더 하이퍼파라미터
        fs: int = 200,
        window_len: int = 200,
        apply_smoothing: bool = False,
        n_components: int = 8,
        hidden_dim: int = 128,

        image_size: int = 384,
        vit: str = 'base',
        vit_grad_ckpt: bool = False,
        vit_ckpt_layer: int = 0,
        embed_dim: int = 256,
    ):
        super().__init__()
        # Visual encoder
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer
        )
        # Sequence encoder
        # EMCSP_EEG_1DCNN_Encoder 기본 인스턴스화
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
        # ITM head: maps concatenated proj dims to 2 classes
        self.itm_head    = layers.Dense(2)

    def call(self, image: tf.Tensor, sequence: tf.Tensor, match_head: str = 'itm'):
        """
        Args:
            image: Tensor of shape (batch, H, W, 3)
            sequence: Tensor of shape (batch, seq_len, channels)
            match_head: 'itm' for matching head, 'itc' for contrastive
        Returns:
            logits for 'itm' or sim matrix for 'itc'
        """
        # Image features
        image_embeds = self.visual_encoder(image)                 # [B, tokens, vision_width]
        img_token    = image_embeds[:, 0, :]                      # [B, vision_width]
        img_feat     = self.vision_proj(img_token)               # [B, embed_dim]

        # Sequence features
        seq_states = self.seq_encoder(sequence)                  # [B, seq_len, hidden_size]
        seq_avg    = tf.reduce_mean(seq_states, axis=1)          # [B, hidden_size]
        seq_feat   = self.seq_proj(seq_avg)                      # [B, embed_dim]

        if match_head == 'itm':
            # Classification: concatenate and predict
            combined = tf.concat([img_feat, seq_feat], axis=1)  # [B, 2*embed_dim]
            logits   = self.itm_head(combined)                  # [B, 2]
            return logits

        elif match_head == 'itc':
            # Contrastive: compute cosine-sim approx via normalized dot
            img_norm = tf.math.l2_normalize(img_feat, axis=1)  # [B, embed_dim]
            seq_norm = tf.math.l2_normalize(seq_feat, axis=1)  # [B, embed_dim]
            sim = tf.matmul(img_norm, seq_norm, transpose_b=True)  # [B, B]
            return sim

        else:
            raise ValueError("match_head must be 'itm' or 'itc'")


def blip_itm(pretrained: str = '', **kwargs) -> BLIP_ITM:
    model = BLIP_ITM(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    return model