# SPDX-License-Identifier: Apache-2.0
# Molmo2 vLLM model implementation for verl-agent.
#
# Adapted from vLLM's molmo.py (Molmo v1) for the Molmo2 architecture.
# Key differences from Molmo v1:
# - Nested config: vit_config, adapter_config, text_config
# - No ViT class token (0 prefix tokens)
# - ViT patch embedding has bias
# - pooled_patches_idx based vision pooling (instead of image_masks)
# - SwiGLU image projector (w1/w3 -> w2)
# - Post-norm decoder layers (norm-after pattern)
# - Qwen3-style per-head QK norm

import math
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from functools import cached_property
from itertools import islice
from typing import Annotated, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig

from vllm.attention import Attention
from vllm.attention.layer import MultiHeadAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItems)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptIndexTargets,
                                        PromptInsertion, PromptUpdate,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings, SupportsMultiModal, SupportsPP)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, WeightsMapper, flatten_bn,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix)


# Constants
ADDITIONAL_VOCAB_SIZE = 128
POOLING_SIZE = 2

# Special token strings
IMAGE_PATCH_TOKEN = "<im_patch>"
IMAGE_LOW_RES_TOKEN = "<im_low>"
IM_COL_TOKEN = "<im_col>"
IM_START_TOKEN = "<im_start>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
IM_END_TOKEN = "<im_end>"
FRAME_START_TOKEN = "<frame_start>"
FRAME_END_TOKEN = "<frame_end>"


# ===== Vision Components =====

class Molmo2ViTMLP(nn.Module):
    """MLP used in Vision Transformer (uses gelu_pytorch_tanh, has bias)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.w1 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
        )
        self.act = nn.GELU(approximate="tanh")
        self.w2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.w1(x)
        x = self.act(x)
        x, _ = self.w2(x)
        return x


class Molmo2ViTMultiHeadDotProductAttention(nn.Module):
    """Multi-head attention used in Vision Transformer and for pooling."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_bias: bool = True,
        input_dim: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()

        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = head_dim

        self.total_num_kv_heads = num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        actual_input_dim = input_dim or hidden_size

        self.wq = ColumnParallelLinear(
            actual_input_dim,
            self.total_num_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wk = ColumnParallelLinear(
            actual_input_dim,
            self.total_num_kv_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wv = ColumnParallelLinear(
            actual_input_dim,
            self.total_num_kv_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wo = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=use_bias,
            quant_config=quant_config,
        )

        self.scale = self.head_dim**-0.5
        self.attn = MultiHeadAttention(self.num_heads,
                                       self.head_dim,
                                       self.scale,
                                       num_kv_heads=self.num_kv_heads)

    def forward(self,
                inputs_q: torch.Tensor,
                inputs_kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        xq, _ = self.wq(inputs_q)
        xk, _ = self.wk(inputs_k)
        xv, _ = self.wv(inputs_v)

        output = self.attn(xq, xk, xv)
        output, _ = self.wo(output)
        return output


class Molmo2VisionBlock(nn.Module):
    """Residual attention block used in Vision Transformer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_norm_eps: float,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.attention = Molmo2ViTMultiHeadDotProductAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            quant_config=quant_config,
        )
        self.feed_forward = Molmo2ViTMLP(
            hidden_size, intermediate_size, quant_config)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Molmo2VisionBlockCollection(nn.Module):
    """Collection of residual attention blocks."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_norm_eps: float,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList([
            Molmo2VisionBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                layer_norm_eps=layer_norm_eps,
                quant_config=quant_config,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            x = r(x)
            hidden_states.append(x)
        return hidden_states


class Molmo2VisionTransformer(nn.Module):
    """Vision Transformer - no class token, bias in patch embedding."""

    def __init__(
        self,
        vit_config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = vit_config.hidden_size
        self.num_prefix_tokens: int = 0  # Molmo2 has no class token

        self.positional_embedding = nn.Parameter(
            torch.zeros(vit_config.image_num_pos, hidden_size))

        image_patch_size = vit_config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            hidden_size,
            bias=True,  # Molmo2 has bias
        )

        self.transformer = Molmo2VisionBlockCollection(
            num_layers=vit_config.num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=vit_config.intermediate_size,
            num_heads=vit_config.num_attention_heads,
            num_kv_heads=vit_config.num_key_value_heads,
            head_dim=vit_config.head_dim,
            layer_norm_eps=vit_config.layer_norm_eps,
            quant_config=quant_config,
        )

        h, w = vit_config.image_default_input_size
        self.patch_num = (h // image_patch_size, w // image_patch_size)

    def add_pos_emb(self, x: torch.Tensor, patch_num: tuple) -> torch.Tensor:
        pos_emb = self.positional_embedding
        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])),
             int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1]))

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb,
                size=(patch_num_0, patch_num_1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + pos_emb[None, :, :].to(x.dtype)
        return x

    def forward(self,
                x: torch.Tensor,
                patch_num: Optional[tuple] = None) -> list[torch.Tensor]:
        if patch_num is None:
            patch_num = self.patch_num

        x = self.patch_embedding(x)
        # No class token prepend for Molmo2
        x = self.add_pos_emb(x, patch_num)
        hidden_states = self.transformer(x)
        return hidden_states


class Molmo2ImageProjectorMLP(nn.Module):
    """SwiGLU projector: w1/w3 gate, w2 down projection."""

    def __init__(
        self,
        input_dim: int,
        intermediate_size: int,
        output_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.merged_linear = MergedColumnParallelLinear(
            input_dim,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            output_dim,
            bias=False,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.merged_linear(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Molmo2VisionBackbone(nn.Module):
    """Full vision backbone with pooled_patches_idx-based pooling."""

    packed_modules_mapping = {"merged_linear": ["gate_proj", "up_proj"]}

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        vit_config = config.vit_config
        adapter_config = config.adapter_config

        # Resolve vit_layers (handle negative indices)
        self.vit_layers = []
        for layer in adapter_config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(layer + vit_config.num_hidden_layers)

        # Only build ViT up to the last needed layer
        last_layer_needed = max(self.vit_layers) + 1
        if last_layer_needed < vit_config.num_hidden_layers:
            actual_vit_config = deepcopy(vit_config)
            actual_vit_config.num_hidden_layers = last_layer_needed
        else:
            actual_vit_config = vit_config

        self.image_vit = Molmo2VisionTransformer(
            actual_vit_config, quant_config=quant_config)
        self.num_prefix_tokens = self.image_vit.num_prefix_tokens

        pool_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
        self.image_pooling_2d = Molmo2ViTMultiHeadDotProductAttention(
            hidden_size=adapter_config.hidden_size,
            num_heads=adapter_config.num_attention_heads,
            num_key_value_heads=adapter_config.num_key_value_heads,
            head_dim=adapter_config.head_dim,
            input_dim=pool_dim,
            quant_config=quant_config,
        )
        self.image_projector = Molmo2ImageProjectorMLP(
            input_dim=adapter_config.hidden_size,
            intermediate_size=adapter_config.intermediate_size,
            output_dim=adapter_config.text_hidden_size,
            quant_config=quant_config,
        )

        self.pooling_attention_mask = adapter_config.pooling_attention_mask

    @property
    def dtype(self) -> torch.dtype:
        return self.image_vit.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.image_vit.patch_embedding.weight.device

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        B, T, N, D = images.shape
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        features = []
        for layer in self.vit_layers:
            features.append(image_features[layer])
        image_features = torch.cat(features, dim=-1)

        if self.num_prefix_tokens > 0:
            image_features = image_features[:, 1:]

        image_features = image_features.view(B, T, N, -1)
        return image_features

    def forward(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_image = images.shape[:2]
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.encode_image(images)

        dim = image_features.shape[-1]
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, -1)

        # Use pooled_patches_idx to arrange features for pooling
        batch_idx = torch.arange(
            pooled_patches_idx.shape[0],
            dtype=torch.long,
            device=pooled_patches_idx.device,
        )
        batch_idx = torch.tile(
            batch_idx.view(batch_size, 1, 1),
            [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]],
        )

        # Gather features: [batch, num_pooled_tokens, pool_dim, dim]
        to_pool = image_features.reshape(batch_size, -1, dim)[
            batch_idx, torch.clip(pooled_patches_idx, 0)
        ]
        to_pool = to_pool * valid.to(self.dtype)[:, :, :, None]
        to_pool = to_pool.reshape([-1, pooled_patches_idx.shape[-1], dim])

        # Compute query as mean of valid patches
        if self.pooling_attention_mask:
            denom = valid.view(-1, to_pool.shape[-2]).float().sum(-1)
            denom = torch.where(denom == 0, 1, denom)
            query = (to_pool.sum(-2, keepdim=True) /
                     denom[:, None, None].to(to_pool.dtype))
        else:
            query = to_pool.mean(-2, keepdim=True)

        pooled_features = self.image_pooling_2d(query, to_pool)
        pooled_features = pooled_features.reshape(
            [batch_size, -1, pooled_features.shape[-1]])

        # MLP projection
        pooled_features = self.image_projector(pooled_features)

        # Return only valid features
        return pooled_features.view(
            -1, pooled_features.shape[-1])[valid_token.flatten()]

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("merged_linear", "gate_proj", 0),
            ("merged_linear", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ===== Text Components =====

class Molmo2Attention(nn.Module):
    """Molmo2 attention with QKV fused proj, optional QK norm, RoPE."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = text_config.num_attention_heads

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = (text_config.num_key_value_heads
                                   or self.total_num_heads)
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = text_config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = text_config.max_position_embeddings
        self.rope_theta = text_config.rope_theta

        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=text_config.qkv_bias,
            quant_config=quant_config,
        )

        # QK norm (Qwen3-style per-head norm)
        self.tp_rank: Optional[int] = None
        self.k_norm: Optional[nn.Module] = None
        self.q_norm: Optional[nn.Module] = None
        self.qk_norm_type: Optional[str] = None
        if getattr(text_config, 'use_qk_norm', False):
            qk_norm_type = getattr(text_config, 'qk_norm_type', 'olmo')
            self.qk_norm_type = qk_norm_type
            if qk_norm_type == "qwen3":
                # Per-head norm
                self.q_norm = RMSNorm(self.head_dim,
                                      eps=text_config.layer_norm_eps)
                self.k_norm = RMSNorm(self.head_dim,
                                      eps=text_config.layer_norm_eps)
            else:
                # Full-dim norm
                self.tp_rank = get_tensor_model_parallel_rank()
                self.q_norm = RMSNorm(
                    self.total_num_heads * self.head_dim,
                    eps=text_config.layer_norm_eps)
                self.k_norm = RMSNorm(
                    self.total_num_kv_heads * self.head_dim,
                    eps=text_config.layer_norm_eps)

        # Rotary embeddings
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def _apply_qk_norm(self, q: torch.Tensor,
                       k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qk_norm_type == "qwen3":
            # Per-head norm: q/k are already shaped as [tokens, num_heads*head_dim]
            # We need to reshape, apply norm per head, reshape back
            num_tokens = q.shape[0]
            q = q.view(num_tokens, -1, self.head_dim)
            k = k.view(num_tokens, -1, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.view(num_tokens, -1)
            k = k.view(num_tokens, -1)
        else:
            if self.tp_size > 1:
                q = tensor_model_parallel_all_gather(q.contiguous())
                k = tensor_model_parallel_all_gather(k.contiguous())
            q = self.q_norm(q)
            k = self.k_norm(k)
            if self.tp_size > 1:
                from functools import partial
                splitter = partial(split_tensor_along_last_dim,
                                   num_partitions=self.tp_size)
                q = splitter(q)[self.tp_rank]
                k = splitter(k)[self.tp_rank]
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Molmo2MLP(nn.Module):
    """SwiGLU MLP with fused gate+up projection."""

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.intermediate_size = text_config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Molmo2DecoderLayer(nn.Module):
    """Pre-norm decoder layer (Molmo2-4B uses norm_after=false)."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        text_config = config.text_config

        self.self_attn = Molmo2Attention(config,
                                         cache_config,
                                         quant_config,
                                         prefix=f"{prefix}.self_attn")
        self.mlp = Molmo2MLP(config, quant_config=quant_config)

        # Pre-norm: input_layernorm is HF's attn_norm, applied before attention
        # post_attention_layernorm is HF's ff_norm, applied before FFN
        self.input_layernorm = RMSNorm(
            text_config.hidden_size, eps=text_config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            text_config.hidden_size, eps=text_config.layer_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm with fused residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Molmo2TextModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = config.text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.embedding_size = text_config.vocab_size + ADDITIONAL_VOCAB_SIZE
        self.embed_tokens = VocabParallelEmbedding(
            self.embedding_size,
            text_config.hidden_size,
            quant_config=quant_config,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config.num_hidden_layers,
            lambda prefix: Molmo2DecoderLayer(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(text_config.hidden_size,
                            text_config.layer_norm_eps)

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], text_config.hidden_size))

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        # For pre-norm, residual is carried through; apply final norm
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ===== Multimodal Processing =====

def _lowest_multiple(x: int, k: int) -> int:
    return (x // k) * k


def get_num_patches(
    num_tiles: int,
    *,
    crop_patches: int,
    left_margin: int,
    right_margin: int,
    pooling_size: int,
) -> int:
    if num_tiles == 1:
        return _lowest_multiple(crop_patches + pooling_size - 1, pooling_size)

    crop_window_patches = crop_patches - (left_margin + right_margin)

    left_num = _lowest_multiple(
        crop_window_patches + left_margin + pooling_size - 1,
        pooling_size,
    )
    middle_num = _lowest_multiple(
        crop_window_patches + pooling_size - 1,
        pooling_size,
    )
    right_num = _lowest_multiple(
        crop_window_patches + right_margin + pooling_size - 1,
        pooling_size,
    )

    return left_num + (num_tiles - 2) * middle_num + right_num


def get_patches_grid_size(
    *,
    tiling_h: int,
    tiling_w: int,
    crop_patches: int,
    left_margin: int,
    right_margin: int,
    pooling_size: int,
) -> tuple[int, int]:
    nrows = get_num_patches(
        tiling_h,
        crop_patches=crop_patches,
        left_margin=left_margin,
        right_margin=right_margin,
        pooling_size=pooling_size,
    )
    ncols = get_num_patches(
        tiling_w,
        crop_patches=crop_patches,
        left_margin=left_margin,
        right_margin=right_margin,
        pooling_size=pooling_size,
    )
    return nrows, ncols


def get_candidate_tilings(max_num: int) -> list[tuple[int, int]]:
    tilings = [(i, j) for i in range(1, max_num + 1)
               for j in range(1, max_num + 1) if i * j <= max_num]
    return sorted(tilings, key=lambda x: x[0] * x[1])


def select_tiling(
    *,
    height: int,
    width: int,
    patch_size: int,
    max_num_patches: int,
):
    tilings = get_candidate_tilings(max_num_patches)
    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size

    original_size = np.array([height, width], dtype=np.float32)
    required_scale_d = candidate_resolutions.astype(
        np.float32) / original_size
    required_scale = required_scale_d.min(axis=-1, keepdims=True)

    if (required_scale < 1).all():
        ix = required_scale.argmax()
    else:
        ix = np.where(required_scale < 1.0, 10e9, required_scale).argmin()

    return candidate_tilings[ix]


class Molmo2ProcessorWrapper:
    """Wraps Molmo2Processor for vLLM's multimodal API."""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def __call__(self, *args, **kwargs):
        """Delegate to the underlying HF processor so this wrapper is callable."""
        return self.processor(*args, **kwargs)

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.processor.tokenizer.vocab

    @cached_property
    def max_crops(self) -> int:
        return self.processor.image_processor.max_crops

    @cached_property
    def base_image_input_size(self) -> tuple[int, int]:
        size = self.processor.image_processor.size
        return (size["height"], size["width"])

    @cached_property
    def image_patch_size(self) -> int:
        return self.processor.image_processor.patch_size

    @cached_property
    def overlap_margins(self) -> tuple[int, int]:
        margins = self.processor.image_processor.overlap_margins
        return margins[0], margins[1]

    @cached_property
    def image_token_length_w(self) -> int:
        # Compute from base_image_input_size, patch_size, and pooling_size
        base_w = self.base_image_input_size[1]
        patches_w = base_w // self.image_patch_size
        pool_w = self.processor.image_processor.pooling_size[1]
        return patches_w // pool_w

    @cached_property
    def image_token_length_h(self) -> int:
        base_h = self.base_image_input_size[0]
        patches_h = base_h // self.image_patch_size
        pool_h = self.processor.image_processor.pooling_size[0]
        return patches_h // pool_h

    @cached_property
    def image_patch_id(self) -> int:
        return self.vocab[IMAGE_PATCH_TOKEN]

    @cached_property
    def im_col_id(self) -> int:
        return self.vocab[IM_COL_TOKEN]

    @cached_property
    def im_start_id(self) -> int:
        return self.vocab[IM_START_TOKEN]

    @cached_property
    def im_end_id(self) -> int:
        return self.vocab[IM_END_TOKEN]

    @cached_property
    def low_res_im_start_id(self) -> int:
        return self.vocab[LOW_RES_IMAGE_START_TOKEN]

    @property
    def pooling_size(self) -> int:
        return POOLING_SIZE

    def select_tiling(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        max_crops = self.max_crops
        left_margin, right_margin = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size

        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d
        tiling_h, tiling_w = select_tiling(
            height=image_height - total_margin_pixels,
            width=image_width - total_margin_pixels,
            patch_size=crop_window_size,
            max_num_patches=max_crops,
        )
        return tiling_w, tiling_h

    def get_patches_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        left_margin, right_margin = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size
        pooling_size = self.pooling_size

        crop_patches = base_image_input_size[0] // base_image_input_d
        tiling_w, tiling_h = self.select_tiling(
            image_height=image_height,
            image_width=image_width,
        )

        nrows, ncols = get_patches_grid_size(
            tiling_h=tiling_h,
            tiling_w=tiling_w,
            crop_patches=crop_patches,
            left_margin=left_margin,
            right_margin=right_margin,
            pooling_size=pooling_size,
        )
        return ncols, nrows


class Molmo2ProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(self, **kwargs) -> Molmo2ProcessorWrapper:
        processor = self.ctx.get_hf_processor(**kwargs)
        return Molmo2ProcessorWrapper(processor)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Molmo2ProcessorWrapper],
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        ncols, nrows = processor.get_patches_grid_size(
            image_width=image_width,
            image_height=image_height,
        )
        pooling_size = processor.pooling_size
        image_token_length_w = processor.image_token_length_w
        image_token_length_h = processor.image_token_length_h

        # Low-res tokens + high-res tokens
        extra = image_token_length_w * image_token_length_h
        joint = ((ncols + 1) // pooling_size) * ((nrows + 1) // pooling_size)
        return extra + joint

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()
        tilings = get_candidate_tilings(processor.max_crops)
        base_h, base_w = processor.base_image_input_size

        largest_feature_size, largest_feature_pinpoint = 0, None
        for wr, hr in tilings:
            width, height = base_w * wr, base_h * hr
            feat_size = self.get_num_image_tokens(
                image_width=width,
                image_height=height,
                processor=processor,
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint


class Molmo2DummyInputsBuilder(BaseDummyInputsBuilder[Molmo2ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # Return non-empty text to avoid Molmo2 HF processor returning
        # float dtype input_ids when text is empty
        return "<|endoftext|>"

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Optional[Mapping[str, object]] = None,
    ) -> MultiModalDataDict:
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)
        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class Molmo2MultiModalProcessor(
        BaseMultiModalProcessor[Molmo2ProcessingInfo]):

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # Molmo2 HF processor does NOT insert image tokens into input_ids
        # unless the text contains an <|image|> placeholder. We handle
        # image token insertion ourselves via _get_prompt_updates.
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_num_crops = hf_inputs.get("image_num_crops", torch.empty(0))
        image_grids = hf_inputs.get("image_grids", torch.empty(0))

        # Compute pooling tokens per image from image_grids
        # image_grids format: [low_h, low_w, high_h, high_w] per image
        if len(image_grids.shape) > 0 and image_grids.shape[0] > 0:
            pooling_tokens_per_image = (
                image_grids[:, 0] * image_grids[:, 1] +
                image_grids[:, 2] * image_grids[:, 3]
            )
        else:
            pooling_tokens_per_image = torch.empty(0, dtype=torch.long)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_crops),
            image_token_pooling=MultiModalFieldConfig.flat_from_sizes(
                "image", pooling_tokens_per_image),
            image_grids=MultiModalFieldConfig.batched("image"),
            image_num_crops=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_token_length_w = processor.image_token_length_w
        image_token_length_h = processor.image_token_length_h
        pooling_size = processor.pooling_size

        img_patch_id = processor.image_patch_id
        img_col_id = processor.im_col_id
        img_start_id = processor.im_start_id
        img_end_id = processor.im_end_id
        low_res_start_id = processor.low_res_im_start_id

        # Low-res row tokens
        extra_row = [img_patch_id] * image_token_length_w + [img_col_id]
        # Low-res block: <low_res_im_start> + rows + <im_end>
        extra_joint = ([low_res_start_id] + extra_row * image_token_length_h +
                       [img_end_id])

        def get_insertion_molmo2(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = processor.get_patches_grid_size(
                image_width=image_size.width,
                image_height=image_size.height,
            )

            # High-res row tokens
            joint_row = ([img_patch_id] * ((ncols + 1) // pooling_size) +
                         [img_col_id])
            joint = ([img_start_id] + joint_row *
                     ((nrows + 1) // pooling_size) + [img_end_id])

            return PromptUpdateDetails.select_token_id(
                extra_joint + joint,
                embed_token_id=img_patch_id,
            )

        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.start(),
                insertion=get_insertion_molmo2,
            )
        ]


# ===== Top-Level Model =====

def _get_weights_with_merged_embedding(
    weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[tuple[str, torch.Tensor]]:
    """Merge wte.embedding + wte.new_embedding into single embed_tokens."""
    embedding_weights = {}
    for name, weight in weights:
        if "wte.embedding" in name:
            embedding_weights["embedding"] = weight
        elif "wte.new_embedding" in name:
            embedding_weights["new_embedding"] = weight
        else:
            yield (name, weight)
    if "embedding" in embedding_weights and "new_embedding" in embedding_weights:
        merged = torch.cat(
            [embedding_weights["embedding"],
             embedding_weights["new_embedding"]],
            dim=0,
        )
        yield ("model.embed_tokens.weight", merged)


@MULTIMODAL_REGISTRY.register_processor(Molmo2MultiModalProcessor,
                                        info=Molmo2ProcessingInfo,
                                        dummy_inputs=Molmo2DummyInputsBuilder)
class Molmo2ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    # Weight mapping: Molmo2 HF → vLLM
    # NOTE: Unlike Molmo v1, Molmo2 HF has att_proj/attn_out/q_norm/k_norm
    # INSIDE self_attn, and ff_proj/ff_out INSIDE mlp. So the substr mapping
    # only needs to rename the leaf names, not add parent module prefixes.
    # q_norm/k_norm don't need mapping since they keep the same name.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # vision projector (gate_proj/up_proj handled by stacked_params)
            "image_projector.w1.": "image_projector.gate_proj.",
            "image_projector.w3.": "image_projector.up_proj.",
            "image_projector.w2.": "image_projector.down_proj.",
            # text backbone attention (inside self_attn already)
            "att_proj": "qkv_proj",
            "attn_out": "o_proj",
            # text backbone MLP (inside mlp already)
            "ff_proj": "gate_up_proj",
            "ff_out": "down_proj",
            # text backbone norms (at decoder layer level)
            # attn_norm = pre-attention norm → input_layernorm
            # ff_norm = pre-FFN norm → post_attention_layernorm
            "attn_norm": "input_layernorm",
            "ff_norm": "post_attention_layernorm",
        },
        orig_to_new_prefix={
            # vision backbone
            "model.vision_backbone.": "vision_backbone.",
            # language backbone
            "model.transformer.blocks.": "model.layers.",
            "model.transformer.ln_f.": "model.norm.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_up_proj"],
        "merged_linear": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_backbone = Molmo2VisionBackbone(
            config, quant_config=quant_config)
        self.model = Molmo2TextModel(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))

        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.logits_processor = LogitsProcessor(
            config.text_config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.img_patch_id = None

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> Optional[dict]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_token_pooling = kwargs.pop("image_token_pooling", None)
        image_grids = kwargs.pop("image_grids", None)
        image_num_crops = kwargs.pop("image_num_crops", None)

        if pixel_values is None:
            return None

        if not isinstance(image_num_crops, (torch.Tensor, list)):
            raise ValueError("Incorrect type of image_num_crops. "
                             f"Got type: {type(image_num_crops)}")
        image_num_crops = flatten_bn(image_num_crops, concat=True)

        return dict(
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
        )

    def _process_image_input(
        self,
        image_input: dict,
    ) -> list[torch.Tensor]:
        pixel_values = image_input["pixel_values"]
        image_token_pooling = image_input["image_token_pooling"]
        image_grids = image_input["image_grids"]
        image_num_crops = image_input["image_num_crops"]

        # Flatten all crops
        pixel_values_flat = flatten_bn(pixel_values, concat=True)
        image_token_pooling_flat = flatten_bn(
            image_token_pooling, concat=True)
        image_grids_flat = flatten_bn(image_grids, concat=True)

        # Build batched images for the vision backbone
        # We process all images as a single batch item
        n_crops = pixel_values_flat.shape[0]
        n_patches = pixel_values_flat.shape[1]

        # Process all crops as single batch
        images = pixel_values_flat.unsqueeze(0)  # [1, n_crops, n_patches, D]
        pooled_patches_idx = image_token_pooling_flat.unsqueeze(
            0)  # [1, n_tokens, pool_dim]

        image_features = self.vision_backbone(
            images=images,
            pooled_patches_idx=pooled_patches_idx,
        )

        # Split features per image based on image_grids
        # Each image produces (resized_h*resized_w + height*width) tokens
        result = []
        offset = 0
        for grid in image_grids_flat:
            resized_h, resized_w, height, width = grid.tolist()
            num_tokens = int(resized_h * resized_w + height * width)
            result.append(image_features[offset:offset + num_tokens])
            offset += num_tokens

        return result

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.model(input_ids,
                                   positions,
                                   intermediate_tensors,
                                   inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        weights = _get_weights_with_merged_embedding(weights)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector="vision_backbone.image_projector",
            tower_model="vision_backbone",
        )
