# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
DeepSeek-V3 model plugin for AWEX weight synchronization.

DeepSeek-V3 uses Multi-head Latent Attention (MLA) which has a different
parameter structure from standard QKV attention:

Megatron parameter names -> HuggingFace parameter names:
- self_attention.linear_q_down_proj.weight -> self_attn.q_a_proj.weight
- self_attention.linear_q_up_proj.weight -> self_attn.q_b_proj.weight
- self_attention.linear_q_up_proj.layer_norm_weight -> self_attn.q_a_layernorm.weight
- self_attention.linear_kv_down_proj.weight -> self_attn.kv_a_proj_with_mqa.weight
- self_attention.linear_kv_up_proj.weight -> self_attn.kv_b_proj.weight
- self_attention.linear_kv_up_proj.layer_norm_weight -> self_attn.kv_a_layernorm.weight
- self_attention.linear_q_proj.weight -> self_attn.q_proj.weight
- self_attention.linear_proj.weight -> self_attn.o_proj.weight
"""

from typing import Dict, List, Tuple

import torch
from transformers import PretrainedConfig

from awex import logging
from awex.converter.mcore_converter import McoreToHFWeightConverter, _process_mcore_pp_name
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.converter.weights_converter import quantize_weight
from awex.sharding.param_sharding import ShardingStrategy, ShardingType, get_default_sharding_dim
from awex.sharding.rank_info import RankInfo

logger = logging.getLogger(__name__)


# MLA-specific sharding dimensions
_mla_parameter_sharding_dimensions = {
    # MLA Q projections
    "q_a_proj.weight": 0,  # Low-rank down projection
    "q_b_proj.weight": 0,  # Expand back up projection
    "q_a_layernorm.weight": 0,
    "q_proj.weight": 0,
    # MLA KV projections
    "kv_a_proj_with_mqa.weight": 0,  # Low-rank KV projection
    "kv_b_proj.weight": 0,  # Expand back KV projection
    "kv_a_layernorm.weight": 0,
    # Output projection
    "o_proj.weight": 1,
    # Router bias
    "e_score_correction_bias": 0,
}


def get_mla_sharding_dim(param_name: str) -> int:
    """Get sharding dimension for MLA parameters."""
    for key, dim in _mla_parameter_sharding_dimensions.items():
        if key in param_name:
            return dim
    return get_default_sharding_dim(param_name)


class DeepSeekV3ShardingStrategy(ShardingStrategy):
    """
    Custom sharding strategy for DeepSeek-V3 with MLA (Multi-head Latent Attention).
    """

    def get_attention_sharding_strategy(self, parameter_name, **kwargs):
        """
        Determine sharding strategy for MLA attention parameters.
        """
        sharding_dim = get_mla_sharding_dim(parameter_name)

        # LayerNorm parameters are not sharded
        if "layernorm" in parameter_name.lower() or "norm" in parameter_name.lower():
            return ShardingType.NO_SHARDING, 0, 1

        if self.enable_dp_attention:
            attn_tp_size = self.rank_info.attn_tp_size
            if attn_tp_size > 1:
                return ShardingType.DP_TP_SHARDING, sharding_dim, attn_tp_size
            else:
                return ShardingType.NO_SHARDING, sharding_dim, 1
        else:
            tp_size = self.rank_info.tp_size
            if tp_size > 1:
                return ShardingType.TP_SHARDING, sharding_dim, tp_size
            else:
                return ShardingType.NO_SHARDING, sharding_dim, 1

    def get_sharding_strategy(self, parameter_name, **kwargs):
        """
        Main entry point to determine sharding strategy.
        """
        # MLA attention parameters
        if any(mla_key in parameter_name for mla_key in [
            "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj",
            "q_a_layernorm", "kv_a_layernorm", "q_proj", "o_proj"
        ]):
            return self.get_attention_sharding_strategy(parameter_name, **kwargs)

        # Router bias - not sharded
        if "e_score_correction_bias" in parameter_name:
            return ShardingType.NO_SHARDING, 0, 1

        # Fall back to parent implementation
        return super().get_sharding_strategy(parameter_name, **kwargs)


class McoreToHFWeightConverterDeepSeekV3(McoreToHFWeightConverter):
    """
    Converter for DeepSeek-V3 Megatron weights to HuggingFace format.
    Handles MLA (Multi-head Latent Attention) parameter conversion.
    Supports FP8 quantization for weight transfer.
    """

    # Parameters that should be quantized to FP8
    _FP8_WEIGHT_PATTERNS = {
        # MLA attention
        "q_a_proj.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_b_proj.weight",
        "o_proj.weight",
        # Dense attention (first few layers use standard Q projection)
        "q_proj.weight",
        # MoE experts and shared experts
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }

    def __init__(
        self, hf_config: PretrainedConfig, rank_info: RankInfo, infer_conf: Dict
    ):
        super().__init__(hf_config, rank_info, infer_conf)
        # DeepSeek-V3 specific config
        self.q_lora_rank = getattr(hf_config, "q_lora_rank", 1536)
        self.kv_lora_rank = getattr(hf_config, "kv_lora_rank", 512)

        # FP8 quantization config (matches Slime's logic)
        self.quantization_config = getattr(hf_config, "quantization_config", None) or {}
        self.quant_method = self.quantization_config.get("quant_method") if self.quantization_config else None
        # weight_block_size from quantization_config, e.g. [128, 128]
        self.weight_block_size = self.quantization_config.get("weight_block_size") if self.quantization_config else None
        if self.quant_method:
            assert self.quant_method == "fp8", f"Only fp8 quantization is supported, got {self.quant_method}"
            logger.info(f"DeepSeekV3 converter: FP8 quantization enabled, weight_block_size={self.weight_block_size}")
        else:
            logger.info("DeepSeekV3 converter: No quantization")

        logger.info(f"DeepSeekV3 converter initialized: q_lora_rank={self.q_lora_rank}, kv_lora_rank={self.kv_lora_rank}")

    def _fuse_qkv(self, name: str) -> bool:
        """DeepSeek-V3 MLA does not use fused QKV."""
        return False

    def _fuse_gate_up_proj(self, name: str) -> bool:
        """Do not fuse gate and up projections."""
        return False

    def _should_fp8_quantize(self, param_name: str) -> bool:
        """
        Determine if a parameter should be quantized to FP8.

        Quantize:
        - MLA attention: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
        - MoE: gate_proj, up_proj, down_proj (experts and shared_experts)

        Skip:
        - LayerNorm, bias, embedding, lm_head
        - Already FP8 scale parameters
        """
        if not self.quant_method:
            return False

        # Skip non-weight parameters
        if not param_name.endswith(".weight"):
            return False

        # Skip embedding and lm_head
        if "embed_tokens" in param_name or "lm_head" in param_name or "norm.weight" in param_name:
            return False

        # Skip LayerNorm
        if "layernorm" in param_name.lower():
            return False

        # Check against FP8 patterns
        return any(pattern in param_name for pattern in self._FP8_WEIGHT_PATTERNS)

    def _apply_fp8_quantization(
        self, converted_params: List[Tuple[str, torch.Tensor]]
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Apply FP8 quantization to converted parameters.

        Uses quantize_weight() from weights_converter.py which is consistent
        with Slime's quantization logic:
        - UE8M0 decision is based on SGLang's should_deepgemm_weight_requant_ue8m0()
        - NOT hardcoded based on parameter names

        For each parameter that should be quantized:
        1. Quantize weight to float8_e4m3fn
        2. Generate weight_scale_inv (block-wise) or weight_scale (per-tensor)
        """
        if not self.quant_method:
            return converted_params

        result = []
        for param_name, param in converted_params:
            if self._should_fp8_quantize(param_name):
                # Use Slime-compatible quantize_weight function
                quantized_pairs = quantize_weight(
                    param_name, param, self.weight_block_size
                )
                result.extend(quantized_pairs)
                logger.debug(
                    f"FP8 quantized: {param_name} shape={param.shape} -> "
                    f"{len(quantized_pairs)} outputs, weight_block_size={self.weight_block_size}"
                )
            else:
                result.append((param_name, param))
        return result

    def _convert_mla_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert MLA attention parameters from Megatron to HuggingFace format."""

        # Q low-rank down projection (a projection)
        if "linear_q_down_proj.weight" in name:
            return [("self_attn.q_a_proj.weight", parameter)]

        # Q expand up projection (b projection)
        elif "linear_q_up_proj.weight" in name:
            return [("self_attn.q_b_proj.weight", parameter)]

        # Q LayerNorm
        elif "linear_q_up_proj.layer_norm_weight" in name:
            return [("self_attn.q_a_layernorm.weight", parameter)]

        # KV low-rank down projection (a projection with MQA)
        elif "linear_kv_down_proj.weight" in name:
            return [("self_attn.kv_a_proj_with_mqa.weight", parameter)]

        # KV expand up projection (b projection)
        elif "linear_kv_up_proj.weight" in name:
            return [("self_attn.kv_b_proj.weight", parameter)]

        # KV LayerNorm
        elif "linear_kv_up_proj.layer_norm_weight" in name:
            return [("self_attn.kv_a_layernorm.weight", parameter)]

        # Standard Q projection (for first few dense layers)
        elif "linear_q_proj.weight" in name:
            return [("self_attn.q_proj.weight", parameter)]

        # Output projection
        elif "linear_proj.weight" in name:
            return [("self_attn.o_proj.weight", parameter)]

        # Standard QKV (for dense layers that don't use MLA)
        elif "linear_qkv" in name:
            return super()._convert_attention_param(name, parameter, layer_number)

        else:
            raise NotImplementedError(f"Unsupported MLA parameter name: {name}")

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Override to handle MLA parameters."""
        # Check if this is an MLA parameter
        mla_patterns = [
            "linear_q_down_proj", "linear_q_up_proj",
            "linear_kv_down_proj", "linear_kv_up_proj",
            "linear_q_proj", "linear_proj"
        ]
        if any(pattern in name for pattern in mla_patterns):
            return self._convert_mla_attention_param(name, parameter, layer_number)

        # Fall back to parent for standard attention
        return super()._convert_attention_param(name, parameter, layer_number)

    def _convert_expert_bias_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> Tuple[str, torch.Tensor]:
        """Convert DeepSeek-V3 expert bias (e_score_correction_bias)."""
        if "expert_bias" in name:
            return ("mlp.gate.e_score_correction_bias", parameter.to(torch.bfloat16))
        else:
            raise NotImplementedError(f"Unsupported bias parameter name: {name}")

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Convert a parameter from Megatron to HuggingFace format.

        If FP8 quantization is enabled, applies block-wise quantization to
        eligible parameters and generates scale_inv tensors.
        """
        name = name.replace("module.", "")
        name = _process_mcore_pp_name(name, self.rank_info, self.hf_config)

        # Direct mappings
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        if name in direct_name_mapping:
            result = [(direct_name_mapping[name], parameter)]
            return self._apply_fp8_quantization(result)

        # LM head
        if "output_layer.weight" in name:
            result = self._convert_lm_head_param(name, parameter)
            return self._apply_fp8_quantization(result)

        # Layer parameters
        name = name.replace("decoder.layers.", "")
        parts = name.split(".", 1)
        if len(parts) < 2:
            raise NotImplementedError(f"Unsupported parameter name format: {name}")

        layer_number, remaining_name = parts

        # Attention parameters (including MLA)
        if "self_attention" in remaining_name:
            result = [
                (f"model.layers.{layer_number}.{param_name}", param)
                for param_name, param in self._convert_attention_param(
                    remaining_name, parameter, layer_number
                )
            ]
            return self._apply_fp8_quantization(result)

        # MLP parameters
        elif "mlp" in remaining_name:
            # Router
            if "mlp.gate.weight" in name or "mlp.router.weight" in name:
                converted_name, param = self._convert_gate(name, parameter)
                result = [(f"model.layers.{layer_number}.{converted_name}", param)]
                return self._apply_fp8_quantization(result)

            # Expert bias
            elif "expert_bias" in name:
                converted_name, param = self._convert_expert_bias_param(
                    name, parameter, layer_number
                )
                result = [(f"model.layers.{layer_number}.{converted_name}", param)]
                return self._apply_fp8_quantization(result)

            # Other MLP params
            result = [
                (f"model.layers.{layer_number}.{param_name}", param)
                for param_name, param in self._convert_mlp_param(
                    remaining_name, parameter, layer_number
                )
            ]
            return self._apply_fp8_quantization(result)

        # Input/post-attention layer norms
        elif "input_layernorm" in remaining_name or "layernorm" in remaining_name:
            if "pre_mlp" in remaining_name:
                result = [(f"model.layers.{layer_number}.post_attention_layernorm.weight", parameter)]
            else:
                result = [(f"model.layers.{layer_number}.input_layernorm.weight", parameter)]
            return self._apply_fp8_quantization(result)

        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")


class SGlangToHFWeightConverterDeepSeekV3(SGlangToHFWeightConverter):
    """
    Converter for DeepSeek-V3 SGLang weights.

    Key handling: SGLang uses fused_qkv_a_proj_with_mqa internally, but HF format
    (and Megatron) uses separate q_a_proj and kv_a_proj_with_mqa. This converter
    splits the fused tensor into views so TransferPlan can match by name, and
    P2P writes directly update the underlying fused tensor.
    """

    def __init__(self, hf_config, rank_info, infer_conf):
        super().__init__(hf_config, rank_info, infer_conf)
        # MLA dimensions from config
        self.q_lora_rank = getattr(hf_config, "q_lora_rank", 1536)
        self.kv_lora_rank = getattr(hf_config, "kv_lora_rank", 512)
        self.qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", 64)
        logger.info(
            f"DeepSeekV3 SGLang converter initialized: "
            f"q_lora_rank={self.q_lora_rank}, kv_lora_rank={self.kv_lora_rank}, "
            f"qk_rope_head_dim={self.qk_rope_head_dim}"
        )

    def _fuse_qkv(self, name: str) -> bool:
        """DeepSeek-V3 MLA does not use fused QKV."""
        return False

    def _fuse_gate_up_proj(self, name: str) -> bool:
        """Do not fuse gate and up projections."""
        return False

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Convert attention parameters, splitting fused_qkv_a_proj_with_mqa into views.

        SGLang stores: fused_qkv_a_proj_with_mqa.weight (2112, 7168)
        HF format has: q_a_proj.weight (1536, 7168) + kv_a_proj_with_mqa.weight (576, 7168)

        We return views of the fused tensor so P2P writes update the original.

        NOTE: FP8 quantization adds auxiliary parameters like weight_scale, weight_scale_inv.
        These must be passed through without splitting.
        """
        # Only split the actual weight tensor, not FP8 scale parameters
        # e.g., fused_qkv_a_proj_with_mqa.weight -> split
        #       fused_qkv_a_proj_with_mqa.weight_scale -> pass through
        if "fused_qkv_a_proj_with_mqa" in name and name.endswith(".weight"):
            # Split the fused tensor into q_a and kv_a views
            # fused shape: (q_lora_rank + kv_lora_rank + qk_rope_head_dim, hidden_size)
            expected_size = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
            if parameter.shape[0] != expected_size:
                logger.warning(
                    f"Unexpected fused_qkv_a_proj_with_mqa shape: {parameter.shape}, "
                    f"expected first dim {expected_size}. Passing through unchanged."
                )
                return super()._convert_attention_param(name, parameter, layer_number)

            q_a_proj = parameter.narrow(0, 0, self.q_lora_rank)
            kv_a_proj = parameter.narrow(0, self.q_lora_rank,
                                         self.kv_lora_rank + self.qk_rope_head_dim)

            # Return both views with HF-style names
            q_a_name = name.replace("fused_qkv_a_proj_with_mqa", "q_a_proj")
            kv_a_name = name.replace("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa")

            logger.debug(
                f"Split fused_qkv_a_proj_with_mqa: {parameter.shape} -> "
                f"q_a_proj {q_a_proj.shape}, kv_a_proj_with_mqa {kv_a_proj.shape}"
            )
            return [(q_a_name, q_a_proj), (kv_a_name, kv_a_proj)]

        # For other attention params (including FP8 scales), use parent implementation
        return super()._convert_attention_param(name, parameter, layer_number)


# Register the model with AWEX
# Support multiple DeepSeek-V3 variants
CONFIG = [
    {
        "model_name": "DeepseekV3ForCausalLM",  # Main DeepSeek-V3
        "sharding_strategy": DeepSeekV3ShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterDeepSeekV3,
        "sglang_converter": SGlangToHFWeightConverterDeepSeekV3,
    },
    {
        "model_name": "DeepseekV32ForCausalLM",  # V3.2 variant
        "sharding_strategy": DeepSeekV3ShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterDeepSeekV3,
        "sglang_converter": SGlangToHFWeightConverterDeepSeekV3,
    },
    {
        "model_name": "DeepseekV3ForCausalLMNextN",  # NextN variant
        "sharding_strategy": DeepSeekV3ShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterDeepSeekV3,
        "sglang_converter": SGlangToHFWeightConverterDeepSeekV3,
    },
]
