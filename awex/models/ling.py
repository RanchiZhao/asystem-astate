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

from typing import Dict, List, Tuple

import torch
from transformers import PretrainedConfig

from awex import logging
from awex.converter.mcore_converter import McoreToHFWeightConverter
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.converter.weights_converter import quantize_weight
from awex.sharding.param_sharding import ShardingStrategy, ShardingType
from awex.sharding.rank_info import RankInfo

logger = logging.getLogger(__name__)


class BailingMoeShardingStrategy(ShardingStrategy):
    """
    Custom sharding strategy for BailingMoeForCausalLM model architecture.
    """

    def get_sharding_strategy(self, parameter_name, **kwargs):
        if self.engine_name == "mcore":
            if "query_key_value" in parameter_name:
                return ShardingType.NO_SHARDING, 0, 1
        return super().get_sharding_strategy(parameter_name, **kwargs)

    def get_embedding_sharding_strategy(self, parameter_name, **kwargs):
        tp_size = self.rank_info.tp_size
        if not self.enable_dp_attention and tp_size > 1:
            return ShardingType.TP_SHARDING, 0, tp_size
        else:
            return ShardingType.NO_SHARDING, 0, 1


class McoreToHFWeightConverterBailingMoe(McoreToHFWeightConverter):
    def __init__(
        self, hf_config: PretrainedConfig, rank_info: RankInfo, infer_conf: Dict
    ):
        super().__init__(hf_config, rank_info, infer_conf)
        # FP8 quantization config (matches Slime's logic)
        self.quantization_config = getattr(hf_config, "quantization_config", None) or {}
        self.quant_method = self.quantization_config.get("quant_method") if self.quantization_config else None
        # weight_block_size from quantization_config, e.g. [128, 128]
        self.weight_block_size = self.quantization_config.get("weight_block_size") if self.quantization_config else None
        self.fp8_weight_keys = set()
        if self.quant_method:
            assert self.quant_method == "fp8", "Only fp8 quantization is supported"
            self.fp8_weight_keys = {
                "up_proj.weight",
                "down_proj.weight",
                "gate_proj.weight",
                "attention.dense.weight",
                "attention.query_key_value.weight",
            }
            logger.info(f"BailingMoe converter: FP8 quantization enabled, weight_block_size={self.weight_block_size}")

    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        super_converted_params = super().convert_param(name, parameter)
        if not self.quant_method:
            return super_converted_params

        pair_list = []
        for param_name, param in super_converted_params:
            apply_fp8 = any(fp8_key in param_name for fp8_key in self.fp8_weight_keys)

            if apply_fp8:
                # Use Slime-compatible quantize_weight function
                # UE8M0 decision is made by should_use_ue8m0() based on SGLang config
                quantized_pairs = quantize_weight(
                    param_name, param, self.weight_block_size
                )
                pair_list.extend(quantized_pairs)
            else:
                pair_list.append((param_name, param))
        return pair_list


class SGlangToHFWeightConverterBailingMoe(SGlangToHFWeightConverter):
    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False


CONFIG = {
    "model_name": "BailingMoeForCausalLM",
    "sharding_strategy": BailingMoeShardingStrategy,
    "mcore_converter": McoreToHFWeightConverterBailingMoe,
    "sglang_converter": SGlangToHFWeightConverterBailingMoe,
}
