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
Qwen3MoE model plugin for AWEX weight synchronization.

Qwen3MoE uses GQA (Grouped Query Attention) with QK-LayerNorm.
This plugin handles the weight conversion between Megatron and SGLang naming conventions.

Megatron parameter names -> HuggingFace/SGLang parameter names:
- self_attention.linear_qkv.weight -> self_attn.qkv_proj.weight (fused QKV)
- self_attention.linear_proj.weight -> self_attn.o_proj.weight
- self_attention.q_layernorm.weight -> self_attn.q_norm.weight
- self_attention.k_layernorm.weight -> self_attn.k_norm.weight
- pre_mlp_layernorm.weight -> post_attention_layernorm.weight
- mlp.experts.linear_fc1.weight{N} -> mlp.experts.{N}.gate_up_proj.weight
- mlp.experts.linear_fc2.weight{N} -> mlp.experts.{N}.down_proj.weight
- mlp.router.weight -> mlp.gate.weight
"""

from typing import Dict, List, Tuple

import torch
from torch import distributed as dist
from transformers import PretrainedConfig

from awex import logging
from awex.converter.mcore_converter import McoreToHFWeightConverter, _process_mcore_pp_name
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.sharding.param_sharding import ShardingStrategy, ShardingType, get_default_sharding_dim
from awex.sharding.rank_info import RankInfo
from awex.util.common import divide

logger = logging.getLogger(__name__)


# Qwen3MoE-specific sharding dimensions
_qwen3moe_parameter_sharding_dimensions = {
    # Attention projections
    "qkv_proj.weight": 0,  # Fused QKV projection, row parallel
    "o_proj.weight": 1,  # Output projection, column parallel
    # QK LayerNorm - not sharded
    "q_norm.weight": -1,
    "k_norm.weight": -1,
    # MoE expert projections
    "gate_up_proj.weight": 0,
    "down_proj.weight": 1,
    "gate_proj.weight": 0,
    "up_proj.weight": 0,
}


def get_qwen3moe_sharding_dim(param_name: str) -> int:
    """Get sharding dimension for Qwen3MoE parameters."""
    for key, dim in _qwen3moe_parameter_sharding_dimensions.items():
        if key in param_name:
            return dim
    return get_default_sharding_dim(param_name)


class Qwen3MoEShardingStrategy(ShardingStrategy):
    """
    Custom sharding strategy for Qwen3MoE with GQA and QK-LayerNorm.
    """

    def get_attention_sharding_strategy(self, parameter_name, **kwargs):
        """
        Determine sharding strategy for Qwen3MoE attention parameters.
        """
        sharding_dim = get_qwen3moe_sharding_dim(parameter_name)

        # LayerNorm parameters (q_norm, k_norm) are not sharded
        if sharding_dim == -1 or "norm" in parameter_name.lower():
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
        # Attention parameters
        if any(key in parameter_name for key in [
            "qkv_proj", "o_proj", "q_norm", "k_norm"
        ]):
            return self.get_attention_sharding_strategy(parameter_name, **kwargs)

        # Fall back to parent implementation
        return super().get_sharding_strategy(parameter_name, **kwargs)


class McoreToHFWeightConverterQwen3MoE(McoreToHFWeightConverter):
    """
    Converter for Qwen3MoE Megatron weights to HuggingFace/SGLang format.

    Key difference from default converter:
    - Produces fused qkv_proj instead of separate q_proj, k_proj, v_proj
    - Handles QK-LayerNorm (q_norm, k_norm)
    - Handles MoE expert naming conventions
    """

    def __init__(
        self, hf_config: PretrainedConfig, rank_info: RankInfo, infer_conf: Dict
    ):
        super().__init__(hf_config, rank_info, infer_conf)
        # Qwen3MoE specific config
        self.num_attention_heads = hf_config.num_attention_heads
        self.num_key_value_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
        self.hidden_size = hf_config.hidden_size
        # Use kv_channels if available (e.g., Qwen3 sets kv_channels=128)
        self.head_dim = getattr(hf_config, "head_dim", None) or (self.hidden_size // self.num_attention_heads)
        self.num_experts = getattr(hf_config, "num_experts", None)
        logger.info(
            f"Qwen3MoE converter initialized: num_heads={self.num_attention_heads}, "
            f"num_kv_heads={self.num_key_value_heads}, head_dim={self.head_dim}, "
            f"num_experts={self.num_experts}"
        )

    def _fuse_qkv(self, name: str) -> bool:
        """Qwen3MoE uses fused QKV projection."""
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        """Fuse gate and up projections for SGLang compatibility."""
        return True

    def _convert_qkv_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Convert Megatron GQA QKV weight layout to SGLang layout.

        Megatron GQA layout (interleaved per KV group):
            [Q_group0, K_group0, V_group0, Q_group1, K_group1, V_group1, ...]

        SGLang layout (consecutive):
            [Q_all, K_all, V_all]

        IMPORTANT: This method converts the LOCAL shard only (no all_gather).
        The TransferPlan handles data routing between different TP configurations.
        """
        from megatron.training import get_args
        from megatron.core import parallel_state as mpu

        args = get_args()
        hidden_size = args.hidden_size
        total_num_heads = args.num_attention_heads
        total_num_kv_heads = args.num_query_groups
        # Use kv_channels if available
        head_size = getattr(args, 'kv_channels', None) or divide(hidden_size, total_num_heads)

        value_num_per_group = divide(total_num_heads, total_num_kv_heads)
        train_tp_size = mpu.get_tensor_model_parallel_world_size()

        # Calculate expected size for LOCAL shard (not global)
        # Each TP rank gets: total_num_kv_heads / train_tp_size groups
        local_num_kv_groups = divide(total_num_kv_heads, train_tp_size)
        q_size_per_group = value_num_per_group * head_size
        kv_size_per_group = head_size
        chunk_size = q_size_per_group + 2 * kv_size_per_group
        expected_local_size = chunk_size * local_num_kv_groups

        actual_size = weight.shape[0]

        if actual_size != expected_local_size:
            logger.warning(
                f"QKV weight size {actual_size} doesn't match expected local GQA size {expected_local_size}. "
                f"tp_size={train_tp_size}, local_kv_groups={local_num_kv_groups}. Returning as-is."
            )
            return weight

        # Split interleaved QKV into separate Q, K, V (LOCAL shard only)
        query_list = []
        key_list = []
        value_list = []

        for group_idx in range(local_num_kv_groups):
            start = group_idx * chunk_size
            q = weight[start : start + q_size_per_group]
            k = weight[start + q_size_per_group : start + q_size_per_group + kv_size_per_group]
            v = weight[start + q_size_per_group + kv_size_per_group : start + chunk_size]
            query_list.append(q)
            key_list.append(k)
            value_list.append(v)

        # Concatenate to SGLang layout [Q, K, V] for LOCAL shard
        all_query = torch.cat(query_list, dim=0)
        all_key = torch.cat(key_list, dim=0)
        all_value = torch.cat(value_list, dim=0)

        result = torch.cat([all_query, all_key, all_value], dim=0)
        logger.debug(
            f"QKV conversion: input_shape={weight.shape}, output_shape={result.shape}, "
            f"tp_rank={mpu.get_tensor_model_parallel_rank()}"
        )
        return result

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert Qwen3MoE attention parameters from Megatron to HuggingFace/SGLang format."""

        # Fused QKV projection
        if "self_attention.linear_qkv.weight" in name:
            converted_weight = self._convert_qkv_weight(parameter)
            return [("self_attn.qkv_proj.weight", converted_weight)]

        # Output projection
        elif "self_attention.linear_proj.weight" in name:
            return [("self_attn.o_proj.weight", parameter)]

        # QK LayerNorm
        elif "self_attention.q_layernorm.weight" in name:
            return [("self_attn.q_norm.weight", parameter)]
        elif "self_attention.k_layernorm.weight" in name:
            return [("self_attn.k_norm.weight", parameter)]

        # Input layernorm (fused with linear_qkv in Megatron)
        elif "self_attention.linear_qkv.layer_norm_weight" in name:
            return [("input_layernorm.weight", parameter)]

        else:
            raise NotImplementedError(f"Unsupported Qwen3MoE attention parameter: {name}")

    def _convert_mlp_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert Qwen3MoE MLP parameters from Megatron to HuggingFace/SGLang format."""

        # Post attention layernorm
        if "pre_mlp_layernorm.weight" in name or "linear_fc1.layer_norm_weight" in name:
            return [("post_attention_layernorm.weight", parameter)]

        # Router
        if "mlp.router.weight" in name:
            return [("mlp.gate.weight", parameter.to(self.router_dtype))]

        # MoE experts
        if "mlp.experts." in name:
            # Extract expert ID: mlp.experts.linear_fc1.weight0 or mlp.experts.local_experts.0.linear_fc1.weight
            if "local_experts" in name:
                # mlp.experts.local_experts.0.linear_fc1.weight
                local_expert_id = int(name.rsplit(".", 3)[-3])
            else:
                # mlp.experts.linear_fc1.weight0
                local_expert_id = int(name.rsplit("weight", 1)[-1])

            num_experts = self.hf_config.num_experts
            num_experts_per_partition = num_experts // self.rank_info.ep_size
            expert_id = local_expert_id + self.rank_info.ep_rank * num_experts_per_partition

            if "linear_fc1" in name:
                # gate_up_proj (fused)
                return [(f"mlp.experts.{expert_id}.gate_up_proj.weight", parameter)]
            elif "linear_fc2" in name:
                return [(f"mlp.experts.{expert_id}.down_proj.weight", parameter)]
            else:
                raise NotImplementedError(f"Unsupported expert param: {name}")

        # Shared expert
        if "shared_experts" in name:
            if "linear_fc1.weight" in name:
                return [("mlp.shared_expert.gate_up_proj.weight", parameter)]
            elif "linear_fc2.weight" in name:
                return [("mlp.shared_expert.down_proj.weight", parameter)]
            elif "gate_weight" in name:
                return [("mlp.shared_expert_gate.weight", parameter)]
            else:
                raise NotImplementedError(f"Unsupported shared expert param: {name}")

        # Dense MLP (non-MoE layers, if any)
        if "linear_fc1.weight" in name:
            return [("mlp.gate_up_proj.weight", parameter)]
        elif "linear_fc2.weight" in name:
            return [("mlp.down_proj.weight", parameter)]

        raise NotImplementedError(f"Unsupported Qwen3MoE MLP parameter: {name}")

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert a Megatron parameter to HuggingFace/SGLang format."""
        name = name.replace("module.", "")
        name = _process_mcore_pp_name(name, self.rank_info, self.hf_config)

        # Direct name mappings
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        if name in direct_name_mapping:
            return [(direct_name_mapping[name], parameter)]

        # LM head
        if "output_layer.weight" in name:
            return [("lm_head.weight", parameter)]

        # Layer-specific parameters
        name = name.replace("decoder.layers.", "")
        layer_number, remaining_name = name.split(".", 1)

        if "self_attention" in remaining_name:
            return [
                (f"model.layers.{layer_number}.{param_name}", param)
                for param_name, param in self._convert_attention_param(
                    remaining_name, parameter, layer_number
                )
            ]
        elif "mlp" in remaining_name or "layernorm" in remaining_name:
            return [
                (f"model.layers.{layer_number}.{param_name}", param)
                for param_name, param in self._convert_mlp_param(
                    remaining_name, parameter, layer_number
                )
            ]
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")


class SGlangToHFWeightConverterQwen3MoE(SGlangToHFWeightConverter):
    """
    Converter for Qwen3MoE SGLang weights to HuggingFace format.
    Handles QK-LayerNorm and fused projections.
    """

    def _fuse_qkv(self, name: str) -> bool:
        """Qwen3MoE uses fused QKV in SGLang."""
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        """SGLang uses fused gate_up_proj for MoE."""
        return True

    def _convert_layer_norm_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert Qwen3MoE layer norm parameters including QK-norm."""
        if "input_layernorm" in name:
            return [(name, parameter)]
        elif "post_attention_layernorm" in name:
            return [(name, parameter)]
        # Qwen3 QK-LayerNorm
        elif "q_norm" in name:
            return [(name, parameter)]
        elif "k_norm" in name:
            return [(name, parameter)]
        else:
            raise NotImplementedError(f"Unsupported layer norm: {name}")


# Register the model plugin
CONFIG = {
    "model_name": "Qwen3MoeForCausalLM",
    "sharding_strategy": Qwen3MoEShardingStrategy,
    "mcore_converter": McoreToHFWeightConverterQwen3MoE,
    "sglang_converter": SGlangToHFWeightConverterQwen3MoE,
}
