import os
import torch.nn as nn
from awq.modules.fused.attn import QuantAttentionFused
import torch
import math
import torch.nn as nn

class MixtralBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        moe,
        norm_1,
        norm_2,
        dev,
        max_seq_len,
        rope_theta,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            use_alibi=False,
            rope_theta=rope_theta,
        ).to(dev)
        self.norm_2 = norm_2.to(dev)
        self.moe = moe
        self.device = dev

    def forward(
        self,
        hidden_states,
        past_key_value,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, past_key_value = self.attn.forward(
            hidden_states=norm_out,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
        )

        h = hidden_states.to(attn_output.device) + attn_output
        out = self.moe.forward(self.norm_2(h))
        out = h + out

        return out, None, past_key_value


class LlamaLikeBlock(nn.Module):
    """
    LlamaLikeBlock is intended to be reused across blocks that have
    an architecture that closely resembles Llama, e.g. Mistral and Aquila.
    """

    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        mlp,
        norm_1,
        norm_2,
        dev,
        max_seq_len,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        use_alibi=False,
        head_dim=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        # To support gemma-7b, its head_dim is separate
        if head_dim:
            self.head_dim = head_dim
        
        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            use_alibi=use_alibi,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            head_dim=head_dim,
        ).to(dev)
        self.norm_2 = norm_2.to(dev)
        self.mlp = mlp.to(dev)
        self.device = dev

        # Interpretability stuff
        self.passes = 0
        self.modifications = 3
        self.p = 1.0
        self.activation_stats = {
            'min': [],
            'max': [],
            'avg': [],
            'count': torch.zeros(self.hidden_size, dtype=torch.long)
        }

    def top_p(self, x, p=0.15):
        if p >= 1:
            return x
        original_dtype = x.dtype
        sorted_logits, sorted_indices = torch.sort(torch.abs(x), descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_keep = cumulative_probs <= p
        
        # Keep at least one token
        sorted_indices_to_keep[..., 0] = True

        # Scatter back to original indexing
        indices_to_keep = sorted_indices_to_keep.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_keep)
        
        # Update activation stats
        self.update_activation_stats(indices_to_keep)

        x = torch.where(indices_to_keep, x, torch.full_like(x, 0))
        return x.to(original_dtype)
        
    def update_activation_stats(self, activations):
        # Flatten all dimensions except the last one (hidden_size)
        flat_activations = activations.view(-1, self.hidden_size)
        
        self.activation_stats['min'].append(flat_activations.min(dim=0).values.min().item())
        self.activation_stats['max'].append(flat_activations.max(dim=0).values.max().item())
        self.activation_stats['avg'].append(flat_activations.float().mean().item())
        self.activation_stats['count'] += flat_activations.sum(dim=0)
    
    def forward(
        self,
        hidden_states,
        past_key_value,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, past_key_value = self.attn.forward(
            hidden_states=norm_out,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
        )
        h = hidden_states.to(attn_output.device) + attn_output
        if self.modifications > self.passes:
            values = self.top_p(h, self.p)
            out = self.norm_2(h)
            out[values==0] = 0
        else:
            out = self.norm_2(h)
        out = h + self.mlp.forward(out)
        self.passes = self.passes + 1
        return out, None, past_key_value
    

class CohereBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        mlp,
        norm_1,
        # norm_2,
        dev,
        max_seq_len,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        use_alibi=False,
        head_dim=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        # To support gemma-7b, its head_dim is separate
        if head_dim:
            self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            use_alibi=use_alibi,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            head_dim=head_dim,
            is_neox=False,
        ).to(dev)
        # self.norm_2 = norm_2.to(dev)
        self.mlp = mlp.to(dev)
        self.device = dev

    def forward(
        self,
        hidden_states,
        past_key_value,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, past_key_value = self.attn.forward(
            hidden_states=norm_out,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
        )

        h = hidden_states.to(attn_output.device) + attn_output
        out = h + self.mlp.forward(norm_out)

        return out, None, past_key_value


class MPTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        qkv_layer,
        o_proj,
        mpt_mlp,
        norm_1,
        norm_2,
        dev,
        max_seq_len,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = 0
        self.hidden_size = hidden_size
        self.norm_1 = norm_1
        self.attn = QuantAttentionFused(
            hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            use_alibi=True,
        ).to(dev)
        self.norm_2 = norm_2
        self.ffn = mpt_mlp.to(dev)
        self.device = dev

    def forward(
        self,
        hidden_states,
        past_key_value,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, past_key_value = self.attn.forward(
            hidden_states=norm_out,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions=False,
            use_cache=True,
        )

        h = hidden_states.to(attn_output.device) + attn_output
        out = h + self.ffn.forward(self.norm_2(h))
        return out, None, past_key_value


class FalconDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        qkv_layer,
        o_proj,
        mlp,
        dev,
        max_seq_len,
        input_layernorm=None,
        ln_attn=None,
        ln_mlp=None,
        new_decoder_arch=True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = 8 if new_decoder_arch else 0
        self.hidden_size = hidden_size
        self.new_decoder_arch = new_decoder_arch

        if new_decoder_arch:
            attention_shapes = None
        else:
            attention_shapes = self._get_attention_shapes(
                n_heads, max_seq_len, self.hidden_size // n_heads
            )

        # TODO: Falcon has ALiBi implemented but which model uses it?
        self.attn = QuantAttentionFused(
            hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            use_alibi=False,
            attention_shapes=attention_shapes,
        ).to(dev)

        if new_decoder_arch:
            self.ln_attn = ln_attn  # before attention
            self.ln_mlp = ln_mlp  # before mlp
        else:
            self.input_layernorm = input_layernorm  # before attention

        self.mlp = mlp
        self.device = dev

    def _get_attention_shapes(self, n_heads, max_seq_len, head_dim):
        batch_size = int(os.getenv("AWQ_BATCH_SIZE", "1"))

        self.attention_shapes = {
            # following fastertransformer definition
            "cache_v": (
                batch_size,
                1,
                max_seq_len,
                head_dim,
            ),
            # 8: pack 8 fp16 in FT, if fp32 then use 4
            "cache_k": (
                batch_size,
                1,
                head_dim // 8,
                max_seq_len,
                8,
            ),
            "xqkv_view": (n_heads + 2, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, :-2],
            "xk_slice": lambda xqkv: xqkv[:, :, [-2]],
            "xv_slice": lambda xqkv: xqkv[:, :, [-1]],
            "xq_view": (n_heads, head_dim),
            "xk_view": (1, head_dim),
            "xv_view": (1, head_dim),
            "xk_reshape": (1, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (1, head_dim),
            "single_xv_view": (1, head_dim),
        }

        return self.attention_shapes

    def forward(
        self,
        hidden_states,
        past_key_value,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
    ):
        if self.new_decoder_arch:
            layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            layernorm_out = self.input_layernorm(hidden_states)

        attn_output, _, past_key_value = self.attn.forward(
            hidden_states=layernorm_out,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions=False,
            use_cache=True,
        )

        h_attn = hidden_states.to(attn_output.device) + attn_output

        if self.new_decoder_arch:
            h_mlp = self.mlp.forward(mlp_layernorm_out)
        else:
            h_mlp = self.mlp.forward(layernorm_out)

        out = h_attn + h_mlp

        return out, None, past_key_value


class Phi3Block(nn.Module):
    """
    Phi3Block is intended to be reused across blocks that have
    an architecture that closely resembles Phi-3.
    """

    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        mlp,
        norm_1,
        norm_2,
        dev,
        max_seq_len,
        rope_theta=10000,
        rope_scaling=None,
        use_alibi=False,
        head_dim=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        # To support models with separate head_dim
        if head_dim:
            self.head_dim = head_dim

        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            use_alibi=use_alibi,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
        ).to(dev)
        self.norm_2 = norm_2.to(dev)
        self.mlp = mlp.to(dev)
        self.device = dev

    def forward(
        self,
        hidden_states,
        past_key_value,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, past_key_value = self.attn.forward(
            hidden_states=norm_out,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
        )

        h = hidden_states.to(attn_output.device) + attn_output
        out = h + self.mlp.forward(self.norm_2(h))

        return out, None, past_key_value