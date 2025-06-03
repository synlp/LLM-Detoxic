# teach方法
from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.get_logger(__name__)

# 重写LlamaModel类

import torch.nn as nn

class LlamaModelTarget(LlamaModel):

    def __init__(self, config:LlamaConfig):
        super().__init__(config=config)
        # 默认不使用注入
        self.alpha = 0
        self.A = None

    def get_teacher(self, teacher):
        self.teacher = teacher

    def use_alignment(self, A):
        self.A = A

    def get_alpha(self, alpha):
        self.alpha = alpha
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # 指导注入
        if self.alpha != 0:
            alpha = self.alpha
            teacher = self.teacher
            A = self.A
            td = teacher.device
            
            teach_emb = teacher(input_ids=input_ids.to(td), attention_mask=attention_mask.to(td), 
                                output_hidden_states=True).hidden_states[-1]
            if A is not None:
                A = A.to(td)
                layer_norm = nn.LayerNorm(4096).to(td)
                teach_emb = layer_norm(torch.matmul(teach_emb, A))

            teach_emb = teach_emb.to(hidden_states.device)
            hidden_states = alpha * teach_emb + (1 - alpha) * hidden_states

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

# 重写LlamaForCausalLM类

class LlamaForCausalLMTarget(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config=config)
        self.model = LlamaModelTarget(config)

    def get_teacher(self, teacher):
        self.model.get_teacher(teacher)
    
    def use_alignment(self, A):
        self.model.use_alignment(A)

    def get_alpha(self, alpha):
        self.model.get_alpha(alpha)