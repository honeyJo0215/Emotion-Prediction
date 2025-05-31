import tensorflow as tf
from tensorflow.keras import Model
from transformers import BertConfig, TFBertModel, TFBertLMHeadModel
from blip import create_vit, init_tokenizer, load_checkpoint
#from JBlip.eeg_1dcnn import Sequence1DEncoder
from EMCSP_1D_CNN import EMCSP_EEG_1DCNN_Encoder
import numpy as np
"""

기존 BLIP 텍스트 모델 수정

 class BLIP_VQA(Model):
     def __init__(self,
                  #med_config='configs/med_config.json',
                  seq_encoder: Model = Sequence1DEncoder(in_channels=1, hidden_size=768, num_layers=4),
                  image_size=480,
                  vit='base',
                  vit_grad_ckpt=False,
                  vit_ckpt_layer=0):
         super().__init__()
         # Visual encoder
         self.visual_encoder, vision_width = create_vit(
             vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
         # self.tokenizer = init_tokenizer()
         # # Text encoder (BERT)
         # encoder_config = BertConfig.from_json_file(med_config)
         # encoder_config.encoder_width = vision_width
         # self.text_encoder = TFBertModel(config=encoder_config, add_pooling_layer=False)
       
         #1D encoder
         if seq_encoder is None:
            raise ValueError("seq_encoder must be provided for non-text modality")
         self.seq_encoder = seq_encoder      # e.g. Sequence1DEncoder(...)
 

         # Text decoder (BERT LM)
         decoder_config = BertConfig.from_json_file(med_config)
         self.text_decoder = TFBertLMHeadModel(config=decoder_config)
     def call(self,
              image,
              question: tf.Tensor, answer=None,
              answer_texts=None,
              n=None,
              weights=None,
              training=False,
              inference='rank',
              k_test=128):
         # Image embeddings
         image_embeds = self.visual_encoder(image)
         batch_size = tf.shape(image_embeds)[0]
         seq_len = tf.shape(image_embeds)[1]
         image_atts = tf.ones((batch_size, seq_len), dtype=tf.int32)

         # Tokenize questions
         # question = self.tokenizer(
         #     question_texts,
         #     padding='longest',
         #     truncation=True,
         #     max_length=35,
         #     return_tensors='tf'
         # )
         # q_input_ids = question['input_ids']
         # 1D 시퀀스 인코더 → last_hidden_state 유사 반환
         # question 은 이제 [batch, seq_len, channels] 의 Tensor
         question_states = self.seq_encoder(question)          # [B, L, H]
         question_atts   = tf.ones(tf.shape(question_states)[:2], dtype=tf.int32)
         # Set first token to encoder token
         q_input_ids = tf.concat([
             tf.fill((batch_size, 1), self.tokenizer.enc_token_id),
             q_input_ids[:, 1:]
         ], axis=1)
         q_atts = question['attention_mask']

         if training:
             # Tokenize answers
             answer = self.tokenizer(
                 answer_texts,
                 padding='longest',
                 return_tensors='tf'
             )
             a_input_ids = answer['input_ids']
             a_input_ids = tf.concat([
                 tf.fill((tf.shape(a_input_ids)[0], 1), self.tokenizer.bos_token_id),
                 a_input_ids[:, 1:]
             ], axis=1)
             a_atts = answer['attention_mask']

             # Prepare targets, mask pad tokens
             a_targets = tf.where(
                 a_input_ids == self.tokenizer.pad_token_id,
                 -100,
                 a_input_ids
             )

             # Encode question
             q_output = self.text_encoder(
                 q_input_ids,
                 attention_mask=q_atts,
                 encoder_hidden_states=image_embeds,
                 encoder_attention_mask=image_atts,
                 return_dict=True
             )
             last_hidden = q_output.last_hidden_state

             # Expand for multiple answers
             question_states = tf.repeat(last_hidden, repeats=n, axis=0)
             question_atts = tf.repeat(q_atts, repeats=n, axis=0)
             # Decode answers
             outputs = self.text_decoder(
                 a_input_ids,
                 attention_mask=a_atts,
                 encoder_hidden_states=question_states,
                 encoder_attention_mask=question_atts,
                 return_dict=True
             )
             logits = outputs.logits  # [batch*n, seq_len, vocab_size]

             # Compute loss
             loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                 from_logits=True,
                 reduction=tf.keras.losses.Reduction.NONE
             )
             token_loss = loss_fn(a_targets, logits)
             mask = tf.cast(a_targets != -100, tf.float32)
             seq_loss = tf.reduce_sum(token_loss * mask, axis=1)
             seq_loss = seq_loss * weights
             loss = tf.reduce_sum(seq_loss) / tf.cast(batch_size, tf.float32)
             return loss

         # Inference
         q_output = self.text_encoder(
             q_input_ids,
             attention_mask=q_atts,
             encoder_hidden_states=image_embeds,
             encoder_attention_mask=image_atts,
             return_dict=True
         )
         if inference == 'generate':
             num_beams = 3
             question_states = tf.repeat(q_output.last_hidden_state, repeats=num_beams, axis=0)
             question_atts = tf.ones(tf.shape(question_states)[:2], dtype=tf.int32)
             model_kwargs = {
                 'encoder_hidden_states': question_states,
                 'encoder_attention_mask': question_atts
             }
             bos_ids = tf.fill((batch_size, 1), self.tokenizer.bos_token_id)
             outputs = self.text_decoder.generate(
                 input_ids=bos_ids,
                 max_length=10,
                 min_length=1,
                 num_beams=num_beams,
                 eos_token_id=self.tokenizer.sep_token_id,
                 pad_token_id=self.tokenizer.pad_token_id,
                 **model_kwargs
             )
             answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
             return answers

         elif inference == 'rank':
             return self.rank_answer(
                 q_output.last_hidden_state,
                 q_atts,
                 answer_ids=tf.convert_to_tensor(answer_texts['input_ids']),
                 answer_atts=tf.convert_to_tensor(answer_texts['attention_mask']),
                 k=k_test
             )
     def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
         num_ques = tf.shape(question_states)[0]
         start_ids = tf.fill((num_ques, 1), answer_ids[0, 0])
         start_outputs = self.text_decoder(
             start_ids,
             encoder_hidden_states=question_states,
             encoder_attention_mask=question_atts,
             return_dict=True
         )
         logits = start_outputs.logits[:, 0, :]
         probs = tf.nn.softmax(logits, axis=1)
         answer_first = answer_ids[:, 1]
         prob_first = tf.gather(probs, answer_first, axis=1)
         topk = tf.math.top_k(prob_first, k=k)
         topk_ids = topk.indices  # [num_ques, k]

         gathered_ids = tf.gather(answer_ids, topk_ids)  # [num_ques, k, seq_len]
         gathered_atts = tf.gather(answer_atts, topk_ids)
         flat_ids = tf.reshape(gathered_ids, [-1, tf.shape(answer_ids)[1]])
         flat_atts = tf.reshape(gathered_atts, [-1, tf.shape(answer_atts)[1]])
         targets = tf.where(flat_ids == self.tokenizer.pad_token_id, -100, flat_ids)

         question_states = tf.repeat(question_states, repeats=k, axis=0)
         question_atts = tf.repeat(question_atts, repeats=k, axis=0)

         outputs = self.text_decoder(
             flat_ids,
             attention_mask=flat_atts,
             encoder_hidden_states=question_states,
             encoder_attention_mask=question_atts,
             return_dict=True
         )
         logits = outputs.logits
         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
             from_logits=True,
             reduction=tf.keras.losses.Reduction.NONE
         )
         token_loss = loss_fn(targets, logits)
         mask = tf.cast(targets != -100, tf.float32)
         seq_loss = tf.reduce_sum(token_loss * mask, axis=1)
         seq_loss = -tf.reshape(seq_loss, (num_ques, k))
         max_idx = tf.argmax(seq_loss, axis=1)
         max_ids = tf.gather(topk_ids, max_idx, batch_dims=1)
         return max_ids
         
"""

class BLIP_VQA(Model):
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

        image_size: int = 480,
        vit: str = 'base',
        vit_grad_ckpt: bool = False,
        vit_ckpt_layer: int = 0,
    ):
        super().__init__()
        # Visual encoder
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1
        )
        # Sequence (EEG/rPPG) encoder
        # if seq_encoder is None:
        #     raise ValueError('seq_encoder must be provided')
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

        # Tokenizer and decoder for answers remain for VQA
        self.tokenizer = init_tokenizer()
        self.text_decoder = TFBertLMHeadModel.from_pretrained('bert-base-uncased')

    def call(
        self,
        image: tf.Tensor,
        question: tf.Tensor,
        answer_texts: list = None,
        n: list = None,
        weights: tf.Tensor = None,
        training: bool = False,
        inference: str = 'rank',
        k_test: int = 128,
    ):
        # Image features
        image_embeds = self.visual_encoder(image)
        batch_size = tf.shape(image_embeds)[0]
        seq_len = tf.shape(image_embeds)[1]
        image_atts = tf.ones((batch_size, seq_len), dtype=tf.int32)
        # Sequence encoder
        question_states = self.seq_encoder(question)  # [B, L_seq, H]
        question_atts = tf.ones(tf.shape(question_states)[:2], dtype=tf.int32)

        if training:
            # Tokenize answer candidates
            ans = self.tokenizer(
                answer_texts,
                padding='longest',
                return_tensors='tf'
            )
            a_ids = ans['input_ids']
            a_atts = ans['attention_mask']
            # Prepend BOS
            bos = tf.fill((tf.shape(a_ids)[0], 1), self.tokenizer.bos_token_id)
            a_input_ids = tf.concat([bos, a_ids[:, 1:]], axis=1)
            # Targets: mask pad
            a_targets = tf.where(
                a_input_ids == self.tokenizer.pad_token_id,
                -100,
                a_input_ids
            )
            # Decode
            outputs = self.text_decoder(
                a_input_ids,
                attention_mask=a_atts,
                encoder_hidden_states=question_states,
                encoder_attention_mask=question_atts,
                labels=a_targets,
                return_dict=True
            )
            token_loss = outputs.loss
            if weights is not None:
                token_loss = token_loss * weights
            return tf.reduce_mean(token_loss)

        # Inference: generate or rank
        if inference == 'generate':
            num_beams = 3
            qs = tf.repeat(question_states, repeats=num_beams, axis=0)
            qa = tf.repeat(question_atts, repeats=num_beams, axis=0)
            bos_ids = tf.fill((batch_size, 1), self.tokenizer.bos_token_id)
            outputs = self.text_decoder.generate(
                input_ids=bos_ids,
                max_length=10,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                encoder_hidden_states=qs,
                encoder_attention_mask=qa,
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Ranking
        # Tokenize answers
        ans = self.tokenizer(
            answer_texts,
            padding='longest',
            return_tensors='tf'
        )
        answer_ids = ans['input_ids']
        answer_atts = ans['attention_mask']
        return self.rank_answer(
            question_states,
            question_atts,
            answer_ids,
            answer_atts,
            k_test
        )

    def rank_answer(
        self,
        question_states: tf.Tensor,
        question_atts: tf.Tensor,
        answer_ids: tf.Tensor,
        answer_atts: tf.Tensor,
        k: int,
    ) -> tf.Tensor:
        num_q = tf.shape(question_states)[0]
        bos = tf.fill((num_q, 1), answer_ids[0, 0])
        start_out = self.text_decoder(
            bos,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True
        )
        logits = start_out.logits[:, 0, :]
        probs = tf.nn.softmax(logits, axis=1)
        first_tok = answer_ids[:, 1]
        first_prob = tf.gather(probs, first_tok, axis=1)
        topk = tf.math.top_k(first_prob, k=k)
        topk_ids = topk.indices
        gathered_ids = tf.gather(answer_ids, topk_ids)
        gathered_atts = tf.gather(answer_atts, topk_ids)
        flat_ids = tf.reshape(gathered_ids, [-1, tf.shape(answer_ids)[1]])
        flat_atts = tf.reshape(gathered_atts, [-1, tf.shape(answer_atts)[1]])
        targets = tf.where(flat_ids == self.tokenizer.pad_token_id, -100, flat_ids)
        qs = tf.repeat(question_states, repeats=k, axis=0)
        qa = tf.repeat(question_atts, repeats=k, axis=0)
        out = self.text_decoder(
            flat_ids,
            attention_mask=flat_atts,
            encoder_hidden_states=qs,
            encoder_attention_mask=qa,
            return_dict=True
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )
        token_loss = loss_fn(targets, out.logits)
        mask = tf.cast(targets != -100, tf.float32)
        seq_loss = tf.reduce_sum(token_loss * mask, axis=1)
        seq_loss = -tf.reshape(seq_loss, (num_q, k))
        idx = tf.argmax(seq_loss, axis=1)
        return tf.gather(topk_ids, idx, batch_dims=1)



def blip_vqa(pretrained='', **kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    return model
