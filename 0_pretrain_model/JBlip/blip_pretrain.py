import tensorflow as tf
from tensorflow.keras import Model, layers
from blip import create_vit
# from JBlip.eeg_1dcnn import Sequence1DEncoder, Sequence1DDecoder
from EMCSP_1D_CNN import EMCSP_EEG_1DCNN_Encoder

# Placeholder for gather operations in distributed setup
@tf.function
def concat_all_gather(tensor):
    return tensor

def all_gather_with_grad(tensors):
    return tensors

class BLIP_Pretrain(Model):
    """
    BLIP Pretrain adapted for 1D signals (EEG/rPPG) instead of text.

    Args:
        seq_encoder: 1D CNN encoder for the signal
        seq_decoder: 1D CNN decoder for reconstruction
        image_size: input image size
        vit: vision transformer variant ('base' or 'large')
        embed_dim: projection dimension for contrastive/ITM
        queue_size: queue length for momentum contrastive learning
        momentum: momentum factor for momentum encoders
    """
    def __init__(
        #self,
        #seq_encoder: Model = Sequence1DEncoder(in_channels=1, hidden_size=768, num_layers=4),
        #seq_decoder: Model = Sequence1DDecoder(hidden_size=768, out_channels=1, num_layers=4),
        self,
        seq_encoder: Model = None,
        # EEG‐CSP-1DCNN 인코더 하이퍼파라미터
        fs: int = 200,
        window_len: int = 200,
        apply_smoothing: bool = False,
        n_components: int = 8,
        hidden_dim: int = 128,

        image_size: int = 224,
        vit: str = 'base',
        vit_grad_ckpt: bool = False,
        vit_ckpt_layer: int = 0,
        embed_dim: int = 256,
        queue_size: int = 57600,
        momentum: float = 0.995,
    ):
        super().__init__()
        # Visual encoder
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.0
        )
        # Sequence encoder and decoder
        #if seq_encoder is None or seq_decoder is None:
        #    raise ValueError("seq_encoder and seq_decoder must be provided")
        
        if seq_encoder is None:
            seq_encoder = EMCSP_EEG_1DCNN_Encoder(
                fs=fs,
                window_len=window_len,
                apply_smoothing=apply_smoothing,
                n_components=n_components,
                hidden_dim=hidden_dim
            )
        self.seq_encoder = seq_encoder
        #self.seq_encoder = seq_encoder
        #self.seq_decoder = seq_decoder
        # Projections for contrastive
        self.vision_proj = layers.Dense(embed_dim)
        self.seq_proj    = layers.Dense(embed_dim)
        # ITM head
        self.itm_head    = layers.Dense(2)
        # Momentum encoders
        self.visual_encoder_m, _ = create_vit(vit, image_size)
        self.vision_proj_m = layers.Dense(embed_dim)
        self.seq_encoder_m = Sequence1DEncoder(
            in_channels=getattr(seq_encoder, 'in_channels', 1),
            hidden_size=getattr(seq_encoder, 'hidden_size', vision_width),
            num_layers=getattr(seq_encoder, 'num_layers', 4)
        )
        self.seq_proj_m    = layers.Dense(embed_dim)
        # Parameter pairs for momentum update
        self.model_pairs = [
            (self.visual_encoder, self.visual_encoder_m),
            (self.vision_proj,     self.vision_proj_m),
            (self.seq_encoder,     self.seq_encoder_m),
            (self.seq_proj,        self.seq_proj_m),
        ]
        self.copy_params()
        # Queues for contrastive
        self.queue_size = queue_size
        self.image_queue = tf.Variable(
            tf.math.l2_normalize(tf.random.normal((embed_dim, queue_size)), axis=0),
            trainable=False
        )
        self.seq_queue = tf.Variable(
            tf.math.l2_normalize(tf.random.normal((embed_dim, queue_size)), axis=0),
            trainable=False
        )
        self.idx_queue = tf.Variable(
            tf.fill((1, queue_size), -100), trainable=False, dtype=tf.int32
        )
        self.ptr_queue = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.momentum = momentum
        self.temp = tf.Variable(0.07, trainable=True, dtype=tf.float32)

    def call(self, image, sequence, alpha, training=False):
        """
        Forward pass returns (loss_ita, loss_itm, loss_rec)
        """
        # Clamp temperature
        self.temp.assign(tf.clip_by_value(self.temp, 0.001, 0.5))
        # Image embeddings
        image_embeds = self.visual_encoder(image, training=training)
        bs = tf.shape(image_embeds)[0]
        image_feat = tf.math.l2_normalize(
            self.vision_proj(image_embeds[:,0,:]), axis=-1
        )
        # Sequence embeddings
        seq_states = self.seq_encoder(sequence)  # [B, L, H]
        # Pool to vector
        seq_feat = tf.math.l2_normalize(
            tf.reduce_mean(self.seq_proj(seq_states), axis=1), axis=-1
        )
        # Contrastive targets
        idx = tf.reshape(tf.range(bs), (-1,1))
        idx_all = tf.concat([tf.transpose(idx), self.idx_queue], axis=1)
        pos = tf.cast(tf.equal(idx, idx_all), tf.float32)
        sim_tgt = pos / tf.reduce_sum(pos, axis=1, keepdims=True)
        # Momentum features
        self._momentum_update()
        image_embeds_m = self.visual_encoder_m(image, training=False)
        image_feat_m   = tf.math.l2_normalize(
            self.vision_proj_m(image_embeds_m[:,0,:]), axis=-1
        )
        seq_states_m   = self.seq_encoder_m(sequence)
        seq_feat_m     = tf.math.l2_normalize(
            tf.reduce_mean(self.seq_proj_m(seq_states_m), axis=1), axis=-1
        )
        # All features for queue
        image_all = tf.concat([tf.transpose(image_feat_m), self.image_queue], axis=1)
        seq_all   = tf.concat([tf.transpose(seq_feat_m),   self.seq_queue],   axis=1)
        # Contrastive logits & losses
        sim_i2t_m = tf.matmul(image_feat_m, seq_all)/self.temp
        sim_t2i_m = tf.matmul(seq_feat_m, image_all)/self.temp
        sim_i2t_targets = alpha*tf.nn.softmax(sim_i2t_m, axis=1) + (1-alpha)*sim_tgt
        sim_t2i_targets = alpha*tf.nn.softmax(sim_t2i_m, axis=1) + (1-alpha)*sim_tgt
        sim_i2t = tf.matmul(image_feat, seq_all)/self.temp
        sim_t2i = tf.matmul(seq_feat,   image_all)/self.temp
        loss_i2t = -tf.reduce_mean(tf.reduce_sum(tf.nn.log_softmax(sim_i2t, axis=1)*sim_i2t_targets, axis=1))
        loss_t2i = -tf.reduce_mean(tf.reduce_sum(tf.nn.log_softmax(sim_t2i, axis=1)*sim_t2i_targets, axis=1))
        loss_ita = (loss_i2t + loss_t2i)/2
        # Enqueue
        self._dequeue_and_enqueue(image_feat_m, seq_feat_m, concat_all_gather(idx))
        # ITM loss
        # Positive pairs
        itm_pos = self.itm_head(tf.concat([image_feat, seq_feat], axis=1))
        # Negative sampling: use simple shuffling
        neg_seq = tf.gather(seq_states, tf.random.shuffle(tf.range(bs)))
        neg_feat = tf.math.l2_normalize(tf.reduce_mean(self.seq_proj(neg_seq),axis=1),axis=-1)
        itm_neg = self.itm_head(tf.concat([image_feat, neg_feat], axis=1))
        logits_itm = tf.concat([itm_pos, itm_neg], axis=0)
        labels_itm = tf.concat([tf.ones(bs), tf.zeros(bs)], axis=0)
        loss_itm = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels_itm, logits_itm, from_logits=True))
        # Reconstruction loss
        #recon = self.seq_decoder(seq_states)
        #loss_rec = tf.reduce_mean(tf.losses.MSE(sequence, recon))
        #return loss_ita, loss_itm, loss_rec
        return loss_ita, loss_itm
    
    @tf.function
    def copy_params(self):
        for src, tgt in self.model_pairs:
            tgt.set_weights(src.get_weights())

    @tf.function
    def _momentum_update(self):
        for src, tgt in self.model_pairs:
            new_w = [self.momentum * tm + (1-self.momentum)*s for s,tm in zip(src.get_weights(), tgt.get_weights())]
            tgt.set_weights(new_w)

    @tf.function
    def _dequeue_and_enqueue(self, img_feat, seq_feat, idxs):
        bs = tf.shape(img_feat)[0]
        ptr = self.ptr_queue
        inds = tf.range(ptr, ptr+bs)[:,None]
        self.image_queue.assign(tf.tensor_scatter_nd_update(self.image_queue, inds, tf.transpose(img_feat)))
        self.seq_queue.assign(tf.tensor_scatter_nd_update(self.seq_queue,   inds, tf.transpose(seq_feat)))
        self.idx_queue.assign(tf.tensor_scatter_nd_update(self.idx_queue, inds, tf.transpose(idxs)))
        self.ptr_queue.assign((ptr+bs)%self.queue_size)


def blip_pretrain(**kwargs) -> BLIP_Pretrain:
    return BLIP_Pretrain(**kwargs)