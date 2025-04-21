import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

@dataclass
class RWKVConfig(PretrainedConfig):
    """
    Configuration class for RWKV model.
    """
    model_type = "rwkv"
    
    def __init__(
        self,
        vocab_size=50277,
        context_length=1024,
        hidden_size=768,
        num_hidden_layers=12,
        attention_hidden_size=None,
        intermediate_size=None,
        layer_norm_epsilon=1e-5,
        bos_token_id=0,
        eos_token_id=0,
        hidden_act="gelu",
        initializer_range=0.02,
        pad_token_id=1,
        tie_word_embeddings=True,
        layerdrop=0.0,
        use_cache=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class RWKVTimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.context_length
        self.dim = config.hidden_size
        
        # TimeMix parameters
        self.time_decay = nn.Parameter(torch.empty(self.dim))
        self.time_first = nn.Parameter(torch.empty(self.dim))
        
        # Projections
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.receptance = nn.Linear(self.dim, self.dim, bias=False)
        self.output = nn.Linear(self.dim, self.dim, bias=False)
        
        # LayerNorm
        self.ln_x = nn.LayerNorm(self.dim, eps=config.layer_norm_epsilon)
        
        # Initialize time_mix params
        with torch.no_grad():
            # Initialize time_decay and time_first based on layer_id
            ratio_1_to_almost0 = 1.0 - (layer_id / config.num_hidden_layers)
            ddd = torch.ones(1, 1, self.dim)
            for i in range(self.dim):
                ddd[0, 0, i] = i / self.dim
                
            # Initialize decay
            decay_speed = torch.ones(self.dim)
            for h in range(self.dim):
                decay_speed[h] = -5 + 8 * (h / (self.dim - 1)) ** (0.7 + 1.3 * ratio_1_to_almost0)
            self.time_decay.data = decay_speed
            
            # Initialize first
            zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.dim)]) * 0.5)
            self.time_first.data = torch.ones(self.dim) * math.log(0.3) + zigzag

    def forward(self, x, state=None):
        # Pre-LayerNorm
        xx = self.ln_x(x)

        # Get sequence length
        B, T, C = xx.size()

        # Calculate K, V, R
        k = self.key(xx)
        v = self.value(xx)
        r = self.receptance(xx)

        # Time-decay values
        time_decay = -torch.exp(self.time_decay)
        time_first = torch.exp(self.time_first)

        # Training mode (parallel processing)
        if state is None:
            new_state = None
            output = torch.zeros_like(r)
            # Process the full sequence with cumulative sum
            for t in range(T):
                # Current token
                kt = k[:, t].unsqueeze(1)  # [B, 1, C]
                vt = v[:, t]               # [B, C]
                rt = r[:, t]               # [B, C]

                if t == 0:
                    # For the first token, simple multiplication
                    wkv = kt.squeeze(1) * time_first
                else:
                    # Fixed version: Correctly reshape for broadcasting
                    decay_mask = torch.exp(time_decay.unsqueeze(0) * torch.arange(t, device=k.device).unsqueeze(-1))
                    k_weighted = k[:, :t] * decay_mask.unsqueeze(0)

                    # Direct sum for correct dimensions
                    wkv_history = torch.sum(k_weighted * v[:, :t], dim=1)  # Shape [B, C]
                    wkv = wkv_history + kt.squeeze(1) * time_first

                # Apply receptance gating
                output[:, t] = torch.sigmoid(rt) * wkv


        else:
            # Training mode (parallel processing)
            new_state = None
            output = torch.zeros_like(r)
            
            # Process the full sequence with cumulative sum
            for t in range(T):
                # Current token
                kt = k[:, t].unsqueeze(1)  # (B, 1, C)
                vt = v[:, t].unsqueeze(2)  # (B, C, 1)
                rt = r[:, t]               # (B, C)
                
                # Calculate attention (without loop for efficiency)
                if t == 0:
                    wkv = (kt * vt) * time_first  # Shape: (B, C, 1)
                else:
                    # Apply time decay to the past tokens
                    decay_mask = torch.exp(time_decay * torch.arange(t, device=k.device)).unsqueeze(0).unsqueeze(-1)
                    k_weighted = k[:, :t] * decay_mask  # (B, t, C)
                    
                    # Weighted sum of keys and values - ensure correct dimensions
                    kv_product = (k_weighted.unsqueeze(-1) * v[:, :t].unsqueeze(2))  # Shape: (B, t, C, 1)
                    wkv = kv_product.sum(dim=1)  # Shape: (B, C, 1)
                    wkv = wkv + (kt * vt) * time_first
                
                # Apply receptance gating - ensure dimensions match
                wkv_flat = wkv.view(B, C)  # Explicitly reshape to (B, C)
                output[:, t] = torch.sigmoid(rt) * wkv_flat
        
        # Output projection
        return self.output(output), new_state


class RWKVChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        
        # Projections
        self.key = nn.Linear(self.dim, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, self.dim, bias=False)
        self.receptance = nn.Linear(self.dim, self.dim, bias=False)
        
        # LayerNorm
        self.ln_x = nn.LayerNorm(self.dim, eps=config.layer_norm_epsilon)
        
        # Initialize with proper gains
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.num_hidden_layers)
            
            # Values from the paper
            self.receptance.weight.data.normal_(mean=0.0, std=0.02 * ratio_1_to_almost0)
            self.key.weight.data.normal_(mean=0.0, std=0.02 * ratio_1_to_almost0)
            self.value.weight.data.normal_(mean=0.0, std=0.02 * ratio_1_to_almost0)

    def forward(self, x):
        # Pre-LayerNorm
        xx = self.ln_x(x)
        
        # Channel mixing with the RWKV-style gating
        k = self.key(xx)
        k = torch.square(torch.relu(k))  # GeGLU-like activation
        kv = self.value(k)
        
        # Apply receptance gating
        return torch.sigmoid(self.receptance(xx)) * kv


class RWKVBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        # Time mixing sub-layer
        self.time_mix = RWKVTimeMix(config, layer_id)
        
        # Channel mixing sub-layer
        self.channel_mix = RWKVChannelMix(config, layer_id)
        
    def forward(self, hidden_states, attention_mask=None, state=None):
        # Get residual
        residual = hidden_states
        
        # Time mixing with residual connection
        time_mix_output, new_state = self.time_mix(hidden_states, state)
        hidden_states = residual + time_mix_output
        
        # Channel mixing with residual connection
        residual = hidden_states
        channel_mix_output = self.channel_mix(hidden_states)
        hidden_states = residual + channel_mix_output
        
        return hidden_states, new_state


class RWKVPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RWKVConfig
    base_model_prefix = "rwkv"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"h.*.time_mix.time_mix_.*", r"lm_head.weight"]
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Default Transformer initialization
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RWKVModel(RWKVPreTrainedModel):
    """
    The bare RWKV Model outputting raw hidden-states.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Token embeddings
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.gradient_checkpointing = False
        
        # RWKV layers
        self.blocks = nn.ModuleList([
            RWKVBlock(config, layer_id=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_input_embeddings(self):
        return self.emb
        
    def set_input_embeddings(self, value):
        self.emb = value
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        states=None,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            inputs_embeds = self.emb(input_ids)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
            
        # Initialize states if not provided
        if states is None:
            states = [None] * len(self.blocks)
        new_states = []
        
        # Process through the layers
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask=attention_mask,
                    state=states[i] if states else None
                )
            else:
                layer_outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    state=states[i] if states else None
                )
            hidden_states = layer_outputs[0]
            new_states.append(layer_outputs[1])
        
        # Final LayerNorm
        hidden_states = self.ln_out(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "hidden_states": all_hidden_states,
                "states": new_states,
            }
        return (hidden_states, all_hidden_states, new_states)


class RWKVForCausalLM(RWKVPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.time_mix.time_mix_.*"]
    
    def __init__(self, config):
        super().__init__(config)
        self.rwkv = RWKVModel(config)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        # if config.tie_word_embeddings:
        #     self.lm_head.weight = self.rwkv.emb.weight
            
        self.gradient_checkpointing = False
            
        # Initialize weights and apply final processing
        self.post_init()
    
    # Enable gradient checkpointing
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RWKVModel):
            module.gradient_checkpointing = value
        
    def get_output_embeddings(self):
        return self.lm_head
        
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
            
        # If generating after prefill, use only the last token
        if past_key_values[0] is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        return {
            "input_ids": input_ids,
            "states": past_key_values,
            "use_cache": kwargs.get("use_cache", self.config.use_cache),
        }
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        states=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        use_cache=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Gradient checkpointing logic
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.rwkv),
                input_ids=input_ids,
                attention_mask=attention_mask,
                states=states,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        else:
            outputs = self.rwkv(
                input_ids=input_ids,
                attention_mask=attention_mask,
                states=states,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        
        # Get the last hidden state
        hidden_states = outputs["last_hidden_state"]
        new_states = outputs["states"] if use_cache else None
        
        # Get logits from the last hidden state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if return_dict:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=new_states,
                hidden_states=outputs.get("hidden_states", None),
                attentions=None,
            )
        return (loss, logits, new_states)
