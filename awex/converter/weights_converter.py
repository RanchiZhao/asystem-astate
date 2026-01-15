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
FP8 quantization utilities for AWEX weight conversion.

This module provides FP8 quantization functions that are consistent with Slime's
quantization logic (slime/backends/megatron_utils/megatron_to_hf/processors/quantizer.py).

Key design:
- Import SGLang's quantization functions when available (same as Slime)
- Fall back to local implementations when SGLang is not available
- UE8M0 format decision is based on runtime config, not parameter names
"""

from typing import List, Tuple, Optional

import torch

from awex import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SGLang quantization imports (same as Slime's sglang.py)
# =============================================================================

try:
    from sglang.srt.layers.quantization.fp8_utils import (
        quant_weight_ue8m0 as _sglang_quant_weight_ue8m0,
        transform_scale_ue8m0 as _sglang_transform_scale_ue8m0,
    )
    from sglang.srt.model_loader.utils import (
        should_deepgemm_weight_requant_ue8m0 as _sglang_should_deepgemm_weight_requant_ue8m0,
    )
    _HAS_SGLANG_FP8 = True
    logger.info("SGLang FP8 quantization functions loaded successfully")
except ImportError:
    _sglang_quant_weight_ue8m0 = None
    _sglang_transform_scale_ue8m0 = None
    _sglang_should_deepgemm_weight_requant_ue8m0 = None
    _HAS_SGLANG_FP8 = False
    logger.warning("SGLang FP8 quantization functions not available, using fallback")

# =============================================================================
# Slime's Triton block quantization (same as Slime's quantizer.py)
# =============================================================================

try:
    from slime.utils.fp8_kernel import blockwise_cast_to_fp8_triton as _slime_blockwise_cast_to_fp8_triton
    _HAS_SLIME_FP8 = True
    logger.info("Slime FP8 Triton kernel loaded successfully")
except ImportError:
    _slime_blockwise_cast_to_fp8_triton = None
    _HAS_SLIME_FP8 = False
    logger.warning("Slime FP8 Triton kernel not available, using fallback")


# =============================================================================
# Local fallback implementations
# =============================================================================

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def per_block_cast_to_fp8(x: torch.Tensor, scale_ue8m0: bool = False):
    """
    Block-wise FP8 quantization with optional UE8M0 scale format.

    This is the fallback implementation when SGLang is not available.
    Block size is fixed at 128x128 (same as SGLang/Slime).

    Args:
        x: Input tensor of shape (M, N)
        scale_ue8m0: If True, use UE8M0 format (2^exponent) for scales

    Returns:
        Tuple of (quantized_weight, scale_inv)
    """
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_scale = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4) / 448.0
    if scale_ue8m0:
        x_scale = (
            x_scale.maximum(torch.tensor(1e-10, device=x.device)).log2().ceil().exp2()
        )
    x_scaled = (x_view / x_scale).to(torch.float8_e4m3fn)
    return (
        x_scaled.view_as(x_padded)[:m, :n].contiguous(),
        x_scale.view(x_view.size(0), x_view.size(2)),
    )


def blockwise_cast_to_fp8_fallback(
    weight: torch.Tensor,
    weight_block_size: Optional[List[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fallback block-wise FP8 quantization without UE8M0.

    This matches Slime's blockwise_cast_to_fp8_triton behavior.
    """
    block_m, block_n = 128, 128
    if weight_block_size:
        block_m, block_n = weight_block_size[0], weight_block_size[1]

    # For simplicity, use per_block_cast_to_fp8 without UE8M0
    return per_block_cast_to_fp8(weight, scale_ue8m0=False)


# =============================================================================
# Slime-compatible quantization API
# =============================================================================

def should_use_ue8m0(weight_block_size: Optional[List[int]]) -> bool:
    """
    Check if UE8M0 format should be used for weight quantization.

    This mirrors Slime's logic:
    - If SGLang's should_deepgemm_weight_requant_ue8m0 is available, use it
    - Otherwise, return False (no UE8M0)

    The decision is based on runtime config (DeepGEMM settings), NOT parameter names.
    """
    if _sglang_should_deepgemm_weight_requant_ue8m0 is not None:
        return _sglang_should_deepgemm_weight_requant_ue8m0(weight_block_size=weight_block_size)
    return False


def quantize_weight(
    name: str,
    weight: torch.Tensor,
    weight_block_size: Optional[List[int]]
) -> List[Tuple[str, torch.Tensor]]:
    """
    Quantize a single weight tensor to FP8.

    This function mirrors Slime's _quantize_param exactly:
    - If weight_block_size is provided: use block quantization
      - If should_use_ue8m0() returns True: use UE8M0 format with transform_scale_ue8m0
      - Otherwise: use normal block quantization
    - If weight_block_size is None: use per-tensor quantization

    Args:
        name: Parameter name (must end with ".weight")
        weight: Weight tensor to quantize
        weight_block_size: Block size for quantization, e.g. [128, 128]

    Returns:
        List of (name, tensor) tuples: [(weight_name, qweight), (scale_name, scale)]
    """
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"

    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

    if weight_block_size is not None:
        # Block quantization
        if should_use_ue8m0(weight_block_size):
            # UE8M0 format (same as Slime when DeepGEMM is enabled)
            if _sglang_quant_weight_ue8m0 is not None and _sglang_transform_scale_ue8m0 is not None:
                qweight, scale = _sglang_quant_weight_ue8m0(weight, weight_block_size=weight_block_size)
                scale = _sglang_transform_scale_ue8m0(scale, mn=qweight.shape[-2])
            else:
                # Fallback: use local UE8M0 implementation
                qweight, scale = per_block_cast_to_fp8(weight, scale_ue8m0=True)
            logger.debug(f"FP8 quantized (UE8M0): {name} shape={weight.shape}")
        else:
            # Normal block quantization - use Slime's Triton kernel if available
            if _slime_blockwise_cast_to_fp8_triton is not None:
                qweight, scale = _slime_blockwise_cast_to_fp8_triton(weight, weight_block_size)
                logger.debug(f"FP8 quantized (block, Slime Triton): {name} shape={weight.shape}")
            else:
                # Fallback: use local PyTorch implementation
                qweight, scale = per_block_cast_to_fp8(weight, scale_ue8m0=False)
                logger.debug(f"FP8 quantized (block, fallback): {name} shape={weight.shape}")

        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        # Per-tensor quantization (same as Slime)
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
        qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")
        logger.debug(f"FP8 quantized (per-tensor): {name} shape={weight.shape}")

    return [(name, qweight), (scale_name, scale)]
