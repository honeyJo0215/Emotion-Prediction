import tensorflow as tf
from tensorflow.keras import Model, layers
#from tensorflow.keras.applications import vit as tfvit  # TF-Keras ViT 모듈
try:
    from tensorflow.keras.applications import vit as tfvit
except ImportError:
    try:
        from vit_keras import vit as tfvit
    except ImportError as e:
        raise ImportError(
            "ViT 모듈을 찾을 수 없습니다. "
            "`pip install --user tensorflow-addons-nightly vit-keras` 후 다시 시도하세요."
        ) from e
from transformers import BertConfig, TFBertModel, TFBertLMHeadModel, BertTokenizer
#from JBlip.blip import create_vit, init_tokenizer
#from JBlip.eeg_1dcnn import Sequence1DEncoder, Sequence1DDecoder
from EMCSP_1D_CNN import EMCSP_EEG_1DCNN_Encoder
import numpy as np
import os
from urllib.parse import urlparse


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    # TF checkpoint loading not implemented
    raise NotImplementedError("load_checkpoint for TF is not implemented.")


# class BLIP_Base(Model):
#     def __init__(
#         self,
#         #med_config='configs/med_config.json',
#         seq_encoder: Model = Sequence1DEncoder(in_channels=1, hidden_size=768, num_layers=4),
#         image_size=224,
#         vit='base',
#         vit_grad_ckpt=False,
#         vit_ckpt_layer=0,
#     ):
#         super().__init__()
#         # Visual encoder
#         self.visual_encoder, vision_width = create_vit(
#             vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.0
#         )
#         # Tokenizer
#         self.tokenizer = init_tokenizer()
#         # # Text encoder
#         # cfg = BertConfig.from_json_file(med_config)
#         # cfg.encoder_width = vision_width
#         # self.text_encoder = TFBertModel(config=cfg, add_pooling_layer=False)

#         #1D encoder
#         if seq_encoder is None:
#             raise ValueError("seq_encoder must be provided for non-text modality")
#         self.seq_encoder = seq_encoder      # e.g. Sequence1DEncoder(...)
 

#     def call(self, image, caption, mode):
#         assert mode in ['image', 'text', 'multimodal'], (
#             "mode parameter must be 'image', 'text', or 'multimodal'"
#         )
#         if mode == 'image':
#             # return image features
#             return self.visual_encoder(image)

#         # tokenize text
#         toks = self.tokenizer(
#             caption,
#             padding='longest',
#             truncation=True,
#             return_tensors='tf'
#         )
#         input_ids = toks['input_ids']
#         att_mask = toks['attention_mask']

#         if mode == 'text':
#             # return text features
#             out = self.text_encoder(
#                 input_ids,
#                 attention_mask=att_mask,
#                 return_dict=True
#             )
#             return out.last_hidden_state

#         # multimodal
#         # image features
#         img_embeds = self.visual_encoder(image)
#         img_atts = tf.ones(tf.shape(img_embeds)[:2], dtype=tf.int32)
#         # set first token to encoder token
#         input_ids = tf.concat([
#             tf.fill((tf.shape(input_ids)[0], 1), self.tokenizer.enc_token_id),
#             input_ids[:, 1:]
#         ], axis=1)
#         out = self.text_encoder(
#             input_ids,
#             attention_mask=att_mask,
#             encoder_hidden_states=img_embeds,
#             encoder_attention_mask=img_atts,
#             return_dict=True
#         )
#         return out.last_hidden_state

class BLIP_Base(Model):
    """
    BLIP Base 모델 - 텍스트 대신 1D 시계열(예: EEG, rPPG) 인코더 사용
    Modes:
      - 'image': 이미지 특징 반환
      - 'text': 시퀀스 특징 반환
      - 'multimodal': (이미지, 시퀀스) 특징 튜플 반환
    """
    def __init__(
        #self,
        #seq_encoder: Model = Sequence1DEncoder,
        
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
        vit_grad_checkpoint: bool = False,
        vit_ckpt_layer: int = 0,
    ):
        super().__init__()
        # Vision encoder
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_checkpoint, vit_ckpt_layer, drop_path_rate=0.0
        )
        # 1D sequence encoder (e.g., EEG/rPPG)
        # if seq_encoder is None:
        #     raise ValueError("seq_encoder must be provided for EEG/R-PPG modality")
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

    def call(self, image: tf.Tensor, sequence: tf.Tensor, mode: str):
        """
        Args:
            image: Tensor of shape (batch, H, W, 3)
            sequence: Tensor of shape (batch, seq_len, channels)
            mode: 'image' | 'text' | 'multimodal'
        Returns:
            image features or sequence features or tuple of both
        """
        assert mode in ['image', 'eeg', 'multimodal'], (
            "mode parameter must be 'image', 'text', or 'multimodal'"
        )
        if mode == 'image':
            # 이미지 특징만
            return self.visual_encoder(image)

        if mode == 'eeg':
            # 시퀀스(EEG/rPPG) 특징만
            return self.seq_encoder(sequence)

        # multimodal: 둘 다 반환
        img_feats = self.visual_encoder(image)
        seq_feats = self.seq_encoder(sequence)
        return img_feats, seq_feats


class BLIP_Decoder(Model):
    def __init__(
        self,
        med_config='configs/med_config.json',
        image_size=384,
        vit='base',
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        prompt='a picture of ',
    ):
        super().__init__()
        # Visual encoder
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.0
        )
        # Tokenizer
        self.tokenizer = init_tokenizer()
        # Text decoder
        cfg = BertConfig.from_json_file(med_config)
        cfg.encoder_width = vision_width
        self.text_decoder = TFBertLMHeadModel(config=cfg)
        self.prompt = prompt
        # compute prompt length (# tokens - 1)
        self.prompt_length = len(self.tokenizer(self.prompt, return_tensors='tf')['input_ids'][0]) - 1

    def call(self, image, caption):
        # LM loss
        img_embeds = self.visual_encoder(image)
        img_atts = tf.ones(tf.shape(img_embeds)[:2], dtype=tf.int32)
        toks = self.tokenizer(
            caption,
            padding='longest',
            truncation=True,
            max_length=40,
            return_tensors='tf'
        )
        input_ids = toks['input_ids']
        att_mask = toks['attention_mask']
        # set BOS token
        input_ids = tf.concat([
            tf.fill((tf.shape(input_ids)[0], 1), self.tokenizer.bos_token_id),
            input_ids[:, 1:]
        ], axis=1)
        # prepare targets
        labels = tf.where(
            input_ids == self.tokenizer.pad_token_id,
            -100,
            input_ids
        )
        # mask out prompt tokens
        prompt_mask = tf.sequence_mask(
            self.prompt_length, maxlen=tf.shape(labels)[1]
        )
        labels = tf.where(prompt_mask, -100, labels)
        # compute loss
        out = self.text_decoder(
            input_ids,
            attention_mask=att_mask,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_atts,
            labels=labels,
            return_dict=True
        )
        return out.loss

    def generate(
        self,
        image,
        sample=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        img_embeds = self.visual_encoder(image)
        if not sample:
            img_embeds = tf.repeat(img_embeds, repeats=num_beams, axis=0)
        img_atts = tf.ones(tf.shape(img_embeds)[:2], dtype=tf.int32)
        model_kwargs = {
            'encoder_hidden_states': img_embeds,
            'encoder_attention_mask': img_atts
        }
        # prepare prompt ids
        batch_size = tf.shape(image)[0]
        prompts = [self.prompt] * int(batch_size)
        toks = self.tokenizer(prompts, return_tensors='tf')
        input_ids = toks['input_ids']
        # set BOS and remove last
        input_ids = tf.concat([
            tf.fill((batch_size, 1), self.tokenizer.bos_token_id),
            input_ids[:, 1:-1]
        ], axis=1)
        if sample:
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs
            )
        else:
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs
            )
        captions = []
        for o in outputs:
            text = self.tokenizer.decode(o, skip_special_tokens=True)
            captions.append(text[len(self.prompt):])
        return captions


def blip_feature_extractor(pretrained='', **kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    return model


def blip_decoder(pretrained='', **kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    return model

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    """
    TensorFlow/Keras 기반 ViT 생성기
    - 'base': ViT-B/16
    - 'large': ViT-L/16
    """
    assert vit in ['base', 'large'], "vit parameter must be 'base' or 'large'"

    if vit == 'base':
        # TF-Keras applications 에서 ViT-B/16 사용
        model = tfvit.ViT(
            include_top=False,
            pretrained='imagenet21k',
            image_size=image_size,
            patch_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            num_layers=12,
            dropout_rate=drop_path_rate,
        )
        vision_width = 768

    else:  # 'large'
        model = tfvit.ViT(
            include_top=False,
            pretrained='imagenet21k',
            image_size=image_size,
            patch_size=16,
            hidden_size=1024,
            mlp_dim=4096,
            num_heads=16,
            num_layers=24,
            dropout_rate=drop_path_rate,
        )
        vision_width = 1024

    # 출력 형태: (batch, num_patches+1, vision_width)
    return model, vision_width