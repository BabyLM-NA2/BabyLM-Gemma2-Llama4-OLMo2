import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class LlamaConfig(PretrainedConfig):
    """
    Configuration class for LLaMA model.
    """
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=None,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_dropout=0.0,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embeddings to q and k tensors."""
    if position_ids is None:
        position_ids = torch.arange(q.size(-2), device=q.device).unsqueeze(0)
    cos = cos.squeeze(0).squeeze(0)
    sin = sin.squeeze(0).squeeze(0)
    cos = cos[position_ids].unsqueeze(-2)
    sin = sin[position_ids].unsqueeze(-2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(orig_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings
        self._build()

    def _build(self):
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(-2)
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len
            self._build()
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.dropout = config.attention_dropout

    def _shape(self, tensor, seq_len, bsz, num_heads):
        return tensor.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, x, mask=None, pos_ids=None, past_kv=None, output_attentions=False, use_cache=False
    ):
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self._shape(q, seq_len, bsz, self.num_heads)
        k = self._shape(k, seq_len, bsz, self.num_key_value_heads)
        v = self._shape(v, seq_len, bsz, self.num_key_value_heads)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        past_kv = (k, v) if use_cache else None
        seq_k = k.size(-2)
        cos, sin = self.rotary_emb(x, seq_k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1).to(q.dtype)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out = self.o_proj(out)
        outputs = (out, past_kv)
        if output_attentions:
            outputs += (attn,)
        return outputs


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.norm1 = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)
        self.norm2 = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, x, mask=None, pos_ids=None, past_kv=None, output_attentions=False, use_cache=False
    ):
        r = x
        x = self.norm1(x)
        attn_outputs = self.self_attn(x, mask, pos_ids, past_kv, output_attentions, use_cache)
        x = r + attn_outputs[0]
        r = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = r + x
        outputs = (x,)
        if use_cache:
            outputs += (attn_outputs[1],)
        if output_attentions:
            outputs += (attn_outputs[2],)
        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None:
            raise ValueError("Must specify input_ids or inputs_embeds")

        bsz, seq_len, _ = inputs_embeds.size()
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = (1.0 - attn_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            attn_mask = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_key_values = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_kv = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = layer(
                hidden_states,
                mask=attn_mask,
                pos_ids=position_ids,
                past_kv=past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_key_values += (layer_outputs[1],)
            if output_attentions:
                all_attentions += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            outputs = (hidden_states, next_key_values, all_hidden_states, all_attentions)
            return outputs
        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=None,
            past_key_values=next_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            return (loss, logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
