import tensorflow as tf
from tensorflow.keras import Model, layers
from transformers import BertTokenizer
from blip import create_vit, load_checkpoint
# from JBlip.eeg_1dcnn import Sequence1DEncoder
from EMCSP_1D_CNN import EMCSP_EEG_1DCNN_Encoder

# Distributed gather placeholder
@tf.function
def concat_all_gather(tensor):
    return tensor

def all_gather_with_grad(tensors):
    return tensors

class BLIP_Retrieval(Model):
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
        queue_size: int = 57600,
        momentum: float = 0.995,
        negative_all_rank: bool = False,
    ):
        super().__init__()
        # Visual encoder
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer
        )
        # Sequence (EEG/rPPG) encoder
        # if seq_encoder is None:
        #     raise ValueError("seq_encoder must be provided for non-text modality")
        # self.seq_encoder = seq_encoder
        if seq_encoder is None:
            seq_encoder = EMCSP_EEG_1DCNN_Encoder(
                fs=fs,
                window_len=window_len,
                apply_smoothing=apply_smoothing,
                n_components=n_components,
                hidden_dim=hidden_dim
            )
        self.seq_encoder = seq_encoder

        # Momentum sequence encoder
        # Clone hyperparameters if available
        in_ch = getattr(seq_encoder, 'in_channels', 1)
        hid  = getattr(seq_encoder, 'hidden_size', vision_width)
        nlay = getattr(seq_encoder, 'num_layers', 4)
        ksz  = getattr(seq_encoder, 'kernel_size', 3)
        self.seq_encoder_m = Sequence1DEncoder(
            in_channels=in_ch, hidden_size=hid, num_layers=nlay, kernel_size=ksz
        )
        # Projects
        self.vision_proj   = layers.Dense(embed_dim)
        self.seq_proj      = layers.Dense(embed_dim)
        self.itm_head      = layers.Dense(2)
        self.vision_proj_m = layers.Dense(embed_dim)
        self.seq_proj_m    = layers.Dense(embed_dim)
        # Momentum encoders pairs
        self.model_pairs = [
            (self.visual_encoder, self.visual_encoder_m),
            (self.vision_proj, self.vision_proj_m),
            (self.seq_encoder,  self.seq_encoder_m),
            (self.seq_proj,     self.seq_proj_m),
        ]
        self.copy_params()
        # Queues
        self.queue_size = queue_size
        self.image_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal((embed_dim, queue_size)), axis=0
            ), trainable=False
        )
        self.text_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal((embed_dim, queue_size)), axis=0
            ), trainable=False
        )
        self.idx_queue = tf.Variable(
            tf.fill((1, queue_size), -100), trainable=False, dtype=tf.int32
        )
        self.ptr_queue = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.momentum = momentum
        self.temp = tf.Variable(0.07, trainable=True, dtype=tf.float32)
        self.negative_all_rank = negative_all_rank
        # Tokenizer for captions/answer texts if needed
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def call(self, image, sequence, alpha, idx, training=False):
        # Clamp temperature
        self.temp.assign(tf.clip_by_value(self.temp, 0.001, 0.5))
        # Image features
        image_embeds = self.visual_encoder(image, training=training)
        bs = tf.shape(image_embeds)[0]
        image_feat = tf.math.l2_normalize(
            self.vision_proj(image_embeds[:, 0, :]), axis=-1
        )
        # Sequence features
        seq_states = self.seq_encoder(sequence)  # [B, L, H]
        seq_feat = tf.math.l2_normalize(
            self.seq_proj(tf.reduce_mean(seq_states, axis=1)), axis=-1
        )
        # Contrastive targets
        idx = tf.reshape(idx, (-1, 1))
        idx_all = tf.concat([tf.transpose(idx), self.idx_queue], axis=1)
        pos_idx = tf.cast(tf.equal(idx, idx_all), tf.float32)
        sim_targets = pos_idx / tf.reduce_sum(pos_idx, axis=1, keepdims=True)
        # Momentum update
        self._momentum_update()
        # Image momentum feat
        image_embeds_m = self.visual_encoder_m(image, training=False)
        image_feat_m = tf.math.l2_normalize(
            self.vision_proj_m(image_embeds_m[:, 0, :]), axis=-1
        )
        image_feat_m_all = tf.concat(
            [tf.transpose(image_feat_m), self.image_queue], axis=1
        )
        # Sequence momentum feat
        seq_states_m = self.seq_encoder_m(sequence)
        seq_feat_m = tf.math.l2_normalize(
            self.seq_proj_m(tf.reduce_mean(seq_states_m, axis=1)), axis=-1
        )
        seq_feat_m_all = tf.concat(
            [tf.transpose(seq_feat_m), self.text_queue], axis=1
        )
        # Similarities
        sim_i2t_m = tf.matmul(image_feat_m, seq_feat_m_all) / self.temp
        sim_t2i_m = tf.matmul(seq_feat_m, image_feat_m_all) / self.temp
        sim_i2t_targets = alpha * tf.nn.softmax(sim_i2t_m, axis=1) + (1 - alpha) * sim_targets
        sim_t2i_targets = alpha * tf.nn.softmax(sim_t2i_m, axis=1) + (1 - alpha) * sim_targets
        sim_i2t = tf.matmul(image_feat, seq_feat_m_all) / self.temp
        sim_t2i = tf.matmul(seq_feat, image_feat_m_all) / self.temp
        loss_i2t = -tf.reduce_mean(
            tf.reduce_sum(tf.nn.log_softmax(sim_i2t, axis=1) * sim_i2t_targets, axis=1)
        )
        loss_t2i = -tf.reduce_mean(
            tf.reduce_sum(tf.nn.log_softmax(sim_t2i, axis=1) * sim_t2i_targets, axis=1)
        )
        loss_ita = (loss_i2t + loss_t2i) / 2
        # Enqueue
        self._dequeue_and_enqueue(image_feat_m, seq_feat_m, concat_all_gather(idx))
        return loss_ita

    @tf.function
    def copy_params(self):
        for src, tgt in self.model_pairs:
            tgt.set_weights(src.get_weights())

    @tf.function
    def _momentum_update(self):
        for src, tgt in self.model_pairs:
            new_w = [
                self.momentum * tm + (1 - self.momentum) * s
                for s, tm in zip(src.get_weights(), tgt.get_weights())
            ]
            tgt.set_weights(new_w)

    @tf.function
    def _dequeue_and_enqueue(self, img_feat, seq_feat, idxs):
        bs = tf.shape(img_feat)[0]
        ptr = self.ptr_queue
        indices = tf.range(ptr, ptr + bs)[:, None]
        self.image_queue.assign(tf.tensor_scatter_nd_update(
            self.image_queue, indices, tf.transpose(img_feat)
        ))
        self.text_queue.assign(tf.tensor_scatter_nd_update(
            self.text_queue, indices, tf.transpose(seq_feat)
        ))
        self.idx_queue.assign(tf.tensor_scatter_nd_update(
            self.idx_queue, indices, tf.transpose(idxs)
        ))
        self.ptr_queue.assign((ptr + bs) % self.queue_size)


def blip_retrieval(pretrained: str = '', **kwargs) -> BLIP_Retrieval:
    model = BLIP_Retrieval(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    return model
