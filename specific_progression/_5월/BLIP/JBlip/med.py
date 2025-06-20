import math
import tensorflow as tf
from tensorflow.keras import layers, activations
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.bert.configuration_bert import BertConfig

# 로깅 설정
import logging
logger = logging.getLogger(__name__)

class BertEmbeddings(layers.Layer):
    """
    단어 임베딩과 위치 임베딩을 결합하여 입력 임베딩을 생성합니다.
    """
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        # 단어 임베딩 레이어
        self.word_embeddings = layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            mask_zero=(config.pad_token_id == 0),
            name="word_embeddings"
        )
        # 위치 임베딩 레이어
        self.position_embeddings = layers.Embedding(
            input_dim=config.max_position_embeddings,
            output_dim=config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="position_embeddings"
        )
        # 레이어 정규화
        self.LayerNorm = layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 드롭아웃
        self.dropout = layers.Dropout(rate=config.hidden_dropout_prob)
        
        # position_ids 버퍼 (고정값)
        self.position_ids = tf.range(start=0, limit=config.max_position_embeddings, dtype=tf.int32)[tf.newaxis, :]
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.config = config

    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, training=False):
        # 입력 시퀀스 길이 확인
        if input_ids is not None:
            seq_length = tf.shape(input_ids)[1]
        else:
            seq_length = tf.shape(inputs_embeds)[1]
        
        # 위치 아이디 생성
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        
        # 단어 임베딩
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        
        # 절대 위치 임베딩 적용
        if self.position_embedding_type == "absolute":
            pos_embeds = self.position_embeddings(position_ids)
            embeddings += pos_embeds
        
        # 정규화 및 드롭아웃
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

class BertSelfAttention(layers.Layer):
    """
    BERT의 멀티-헤드 셀프 어텐션 구현
    """
    def __init__(self, config: BertConfig, is_cross_attention=False, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size가 num_attention_heads의 배수가 아닙니다.")
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # query, key, value 프로젝션 레이어
        self.query = layers.Dense(self.all_head_size, name="query")
        if is_cross_attention:
            self.key = layers.Dense(self.all_head_size, name="key")
            self.value = layers.Dense(self.all_head_size, name="value")
        else:
            self.key = layers.Dense(self.all_head_size, name="key")
            self.value = layers.Dense(self.all_head_size, name="value")
        
        # 드롭아웃
        self.dropout = layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = layers.Embedding(
                input_dim=2 * config.max_position_embeddings - 1,
                output_dim=self.attention_head_size,
                name="distance_embedding"
            )
        self.save_attention = False

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        # (batch, seq_len, all_head_size) -> (batch, num_heads, seq_len, head_size)
        new_shape = tf.concat([tf.shape(x)[:-1], [self.num_attention_heads, self.attention_head_size]], axis=0)
        x = tf.reshape(x, new_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        training=False
    ):
        # Query 계산
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        # Key, Value 계산
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)

        # 어텐션 스코어 계산
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores += attention_mask

        # 소프트맥스 -> 확률
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        
        # 헤드 마스크 적용
        if head_mask is not None:
            attention_probs *= head_mask

        # 컨텍스트 계산
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, tf.shape(hidden_states))

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs

class BertSelfOutput(layers.Layer):
    """
    Self-attention 이후의 출력 처리 (Dense -> Dropout -> Residual + LayerNorm)
    """
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(config.hidden_size, name="dense")
        self.LayerNorm = layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(layers.Layer):
    """
    BertLayer 내 self-attention + self-output 구성
    """
    def __init__(self, config: BertConfig, is_cross_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.self = BertSelfAttention(config, is_cross_attention, name="self")
        self.output = BertSelfOutput(config, name="output")
        self.pruned_heads = set()

    def prune_heads(self, heads):
        # TensorFlow 구현에서는 pruning 미지원
        logger.warning("prune_heads는 TensorFlow에서 지원되지 않습니다.")

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        training=False
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training
        )
        attention_output = self_outputs[0]
        outputs = self_outputs[1:]
        attention_output = self.output(attention_output, hidden_states, training=training)
        return (attention_output,) + tuple(outputs)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import TruncatedNormal
from transformers.activations_tf import get_tf_activation
from transformers.modeling_tf_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
)


class BertIntermediate(layers.Layer):
    """
    중간 크기로 투영하고 활성화 함수를 적용하는 레이어
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 선형 변환 레이어
        self.dense = layers.Dense(
            config.intermediate_size,
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            name="intermediate_dense",
        )
        # 활성화 함수 설정
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):  # hidden_states: (batch, seq_length, hidden_size)
        # 선형 변환 수행
        hidden_states = self.dense(hidden_states)
        # 활성화 함수 적용
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(layers.Layer):
    """
    중간 출력 결과를 다시 숨겨진 크기로 투영하고 LayerNorm, Dropout 적용
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 선형 변환 레이어
        self.dense = layers.Dense(
            config.hidden_size,
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            name="output_dense",
        )
        # 레이어 정규화
        self.LayerNorm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps,
            name="output_layer_norm",
        )
        # 드롭아웃
        self.dropout = layers.Dropout(config.hidden_dropout_prob, name="output_dropout")

    def call(self, hidden_states, input_tensor):
        # 선형 변환
        hidden_states = self.dense(hidden_states)
        # 드롭아웃
        hidden_states = self.dropout(hidden_states)
        # skip connection 및 LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(layers.Layer):
    """
    BERT의 한 층: Self-Attention, (Cross-Attention), FeedForward
    """
    def __init__(self, config, layer_num, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # Self-Attention 레이어 (TensorFlow 버전 필요)
        self.attention = BertAttention(config, name=f"layer_{layer_num}_self_attention")
        # Cross-Attention 레이어 (디코더 모드일 때)
        if config.add_cross_attention:
            self.crossattention = BertAttention(
                config, is_cross_attention=True,
                name=f"layer_{layer_num}_cross_attention",
            )
        # FeedForward 단계
        self.intermediate = BertIntermediate(config, name=f"layer_{layer_num}_intermediate")
        self.output = BertOutput(config, name=f"layer_{layer_num}_output")

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        mode="multimodal",
        training=False,
    ):
        # Self-Attention
        self_attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
            training=training,
        )
        attention_output = self_attn_outputs[0]
        present_key_value = self_attn_outputs[-1]
        all_attentions = self_attn_outputs[1:-1] if output_attentions else []

        # Cross-Attention (디코더 모드)
        if mode == "multimodal" and self.config.add_cross_attention:
            assert encoder_hidden_states is not None, \
                "encoder_hidden_states must be provided for cross-attention"
            cross_attn_outputs = self.crossattention(
                attention_output,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            attention_output = cross_attn_outputs[0]
            all_attentions += cross_attn_outputs[1:-1]

        # FeedForward
        layer_output = self.output(
            self.intermediate(attention_output), attention_output,
        )

        outputs = (layer_output,) + tuple(all_attentions) + (present_key_value,)
        return outputs


class BertEncoder(layers.Layer):
    """
    BERT 인코더: 여러 층을 쌓아서 입력을 처리
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.layer = [BertLayer(config, i, name=f"layer_{i}")
                      for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        mode="multimodal",
        training=False,
    ):
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        next_decoder_cache = []
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            past_key = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key,
                output_attentions=output_attentions,
                mode=mode,
                training=training,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions.append(layer_outputs[1])
            next_decoder_cache.append(layer_outputs[-1])

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            outputs = (hidden_states, next_decoder_cache)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            return outputs

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=tuple(next_decoder_cache),
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
            cross_attentions=None,
        )


class BertPooler(layers.Layer):
    """
    시퀀스 첫 번째 토큰을 pool하여 전체 문장 표현 생성
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(
            config.hidden_size,
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            name="pooler_dense",
        )
        self.activation = tf.keras.activations.tanh

    def call(self, hidden_states):
        # 첫 번째 토큰 ([CLS]) 선택
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(layers.Layer):
    """
    MLM 헤드용 트랜스폼: 선형 -> 활성화 -> LayerNorm
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(
            config.hidden_size,
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            name="pred_transform_dense",
        )
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps,
            name="pred_transform_layer_norm",
        )

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(layers.Layer):
    """
    MLM 헤드: 예측 변환 및 decoder (vocab output)
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.transform = BertPredictionHeadTransform(config, name="pred_head_transform")
        # decoder: hidden_size -> vocab_size
        self.decoder = layers.Dense(
            config.vocab_size,
            use_bias=False,
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            name="pred_decoder",
        )
        # bias
        self.bias = self.add_weight(
            shape=(config.vocab_size,), initializer="zeros", name="pred_bias"
        )

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states + self.bias


class BertOnlyMLMHead(layers.Layer):
    """
    전체 시퀀스에 대한 MLM 예측 점수 반환
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.predictions = BertLMPredictionHead(config, name="mlm_predictions")

    def call(self, sequence_output):
        return self.predictions(sequence_output)


class BertPreTrainedModel(tf.keras.Model):
    """
    가중치 초기화 및 config 관리용 베이스 클래스
    """
    config_class = None
    base_model_prefix = "bert"

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def init_weights(self):
        # 모든 레이어를 수동 초기화하려면 여기서 호출
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                for var in layer.trainable_variables:
                    if 'kernel' in var.name:
                        var.assign(tf.random.normal(var.shape, stddev=self.config.initializer_range))
                    elif 'bias' in var.name:
                        var.assign(tf.zeros(var.shape))


class BertModel(BertPreTrainedModel):
    """
    BERT 기본 모델: embeddings + encoder + pooler
    """
    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(config, **kwargs)
        # 임베딩 레이어 (TensorFlow 버전 필요)
        self.embeddings = BertEmbeddings(config, name="embeddings")
        self.encoder = BertEncoder(config, name="encoder")
        self.pooler = BertPooler(config, name="pooler") if add_pooling_layer else None

    def call(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        return_dict=True,
        mode="multimodal",
        training=False,
    ):
        # 임베딩
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training,
        )
        # 인코더
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
            mode=mode,
            training=training,
        )
        sequence_output = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertLMHeadModel(BertPreTrainedModel):
    """
    MLM을 위한 LM 헤드 모델
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.bert = BertModel(config, add_pooling_layer=False, name="bert")
        self.cls = BertOnlyMLMHead(config, name="cls")

    def call(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        return_dict=True,
        mode="multimodal",
        training=False,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            return_dict=True,
            mode=mode,
            training=training,
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)
        loss = None
        if labels is not None:
            # shift labels and predictions
            shift_logits = prediction_scores[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )
            loss = loss_fn(tf.reshape(shift_labels, [-1]), tf.reshape(shift_logits, [-1, self.config.vocab_size]))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TFCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

