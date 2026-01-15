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
Mock TransferPlan Tests for DeepSeek-V3 Scenarios

本测试文件验证TransferPlan在PP/EP不匹配场景下的正确性，无需真实GPU。

测试场景模拟 DeepSeek-V3 配置:
- 训练: TP=8, PP=4, CP=4, EP=32, 总计 128 GPU
- 推理: TP=8, DP=8, EP=64, 总计 64 GPU (2个实例，每个32卡)

核心验证点:
1. PP mismatch: 训练PP=4 → 推理PP=1，层级需要重组
2. EP mismatch: 训练EP=32 → 推理EP=64，专家需要切分
3. CP handling: CP副本应该被正确识别，提供多源P2P
4. Overlap region: 确保所有推理分片都有对应的训练源
5. Data integrity: 模拟传输后数据值正确

增量更新日志:
- 2026-01-09: 初始版本，PP/EP/CP mismatch 测试
- 2026-01-09: 添加数据完整性验证测试，修正 num_infer_engines=2
"""

from typing import List, Tuple

import pytest
import torch

from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.transfer.transfer_plan import (
    CommunicationOperation,
    TransferPlanBuilder,
    slice_tensor,
)


# ==============================================================================
# Helper Functions for Creating Mock Metadata
# ==============================================================================


def create_shard_meta(
    name: str,
    shape: Tuple[int, ...],
    global_offset: Tuple[int, ...],
    global_rank: int,
    tp_rank: int = 0,
    pp_rank: int = 0,
    ep_rank: int = 0,
    sharding_type: ShardingType = ShardingType.TP_SHARDING,
    sharding_dim: int = 0,
    num_shards: int = 1,
) -> ParameterShardMeta:
    """创建单个分片的元数据"""
    numel = 1
    for dim in shape:
        numel *= dim

    return ParameterShardMeta(
        name=name,
        tp_rank=tp_rank,
        attn_tp_rank=tp_rank,
        pp_rank=pp_rank,
        ep_rank=ep_rank,
        ep_tp_rank=ep_rank * 8 + tp_rank if ep_rank > 0 else tp_rank,
        global_rank=global_rank,
        world_size=128,  # 训练128卡
        engine_rank=0,
        shape=shape,
        numel=numel,
        dtype=torch.bfloat16,
        global_offset=global_offset,
        sharding_type=sharding_type,
        num_shards=num_shards,
        sharding_dim=sharding_dim,
    )


def create_parameter_meta(
    name: str,
    shards: List[ParameterShardMeta],
    global_shape: Tuple[int, ...],
) -> ParameterMeta:
    """创建参数的元数据，包含所有分片和副本信息"""
    global_numel = 1
    for dim in global_shape:
        global_numel *= dim

    # 按 (pp_rank, tp_rank) 分组构建 replica
    # 简化：假设每个唯一的 pp_rank 是一个 replica
    replica_groups = {}
    for shard in shards:
        key = shard.pp_rank
        if key not in replica_groups:
            replica_groups[key] = []
        replica_groups[key].append(shard)

    replicas = [
        ParameterReplicaMeta(shards=sorted(group, key=lambda s: s.tp_rank))
        for group in replica_groups.values()
    ]

    return ParameterMeta(
        name=name,
        global_numel=global_numel,
        global_shape=global_shape,
        dtype=torch.bfloat16,
        shards=shards,
        replicas=replicas,
    )


# ==============================================================================
# Test Case 1: PP Mismatch (PP=4 → PP=1)
# ==============================================================================


class TestPPMismatch:
    """
    测试 Pipeline Parallel 不匹配场景

    训练端: PP=4, 每个PP stage有不同的层
    推理端: PP=1, 所有层在同一个进程

    关键: 推理端需要从4个不同的PP rank获取权重
    """

    def test_pp_mismatch_layer_distribution(self):
        """
        测试PP不匹配时的层分布

        场景:
        - 模型有4层 (简化)
        - 训练: PP=4, 每个PP rank有1层
        - 推理: PP=1, 所有4层在一个进程

        期望: TransferPlan生成4个通信操作，每个PP rank发送1层
        """
        # 简化配置: TP=2, PP=4, 无EP
        # 训练 world_size = 2 * 4 = 8
        # 推理 world_size = 2 (TP=2, PP=1)

        builder = TransferPlanBuilder(
            infer_world_size=2,  # 推理2卡
            train_world_size=8,  # 训练8卡
            num_infer_engines=1,
        )

        # 每层的权重形状: (hidden, hidden) = (256, 256)
        hidden_size = 256
        tp_size = 2
        shard_size = hidden_size // tp_size  # 128

        # 构建训练端元数据: 4层，每层在不同PP rank
        train_metas = []
        for layer_idx in range(4):
            pp_rank = layer_idx  # layer 0 -> pp0, layer 1 -> pp1, ...
            shards = []
            for tp_rank in range(tp_size):
                global_rank = pp_rank * tp_size + tp_rank
                shard = create_shard_meta(
                    name=f"layers.{layer_idx}.mlp.weight",
                    shape=(shard_size, hidden_size),  # TP切分第0维
                    global_offset=(tp_rank * shard_size, 0),
                    global_rank=global_rank,
                    tp_rank=tp_rank,
                    pp_rank=pp_rank,
                    sharding_type=ShardingType.TP_SHARDING,
                    sharding_dim=0,
                    num_shards=tp_size,
                )
                shards.append(shard)

            meta = create_parameter_meta(
                name=f"layers.{layer_idx}.mlp.weight",
                shards=shards,
                global_shape=(hidden_size, hidden_size),
            )
            train_metas.append(meta)

        # 构建推理端元数据: 所有4层在PP=0
        infer_metas = []
        for layer_idx in range(4):
            shards = []
            for tp_rank in range(tp_size):
                global_rank = tp_rank  # 推理端只有TP，没有PP
                shard = create_shard_meta(
                    name=f"layers.{layer_idx}.mlp.weight",
                    shape=(shard_size, hidden_size),
                    global_offset=(tp_rank * shard_size, 0),
                    global_rank=global_rank,
                    tp_rank=tp_rank,
                    pp_rank=0,  # 推理端PP=0
                    sharding_type=ShardingType.TP_SHARDING,
                    sharding_dim=0,
                    num_shards=tp_size,
                )
                shards.append(shard)

            meta = create_parameter_meta(
                name=f"layers.{layer_idx}.mlp.weight",
                shards=shards,
                global_shape=(hidden_size, hidden_size),
            )
            infer_metas.append(meta)

        # 构建传输计划
        operations = builder.build_weights_mapping_operations(
            infer_metas, train_metas
        )

        # 验证: 应该有 4层 * 2TP = 8 个操作
        assert len(operations) == 8, f"Expected 8 operations, got {len(operations)}"

        # 验证: 每层的操作来自正确的PP rank
        for op in operations:
            # 解析层号
            param_name = op.send_shard_meta.name
            layer_idx = int(param_name.split(".")[1])

            # 训练端的send_rank应该对应正确的PP rank
            expected_pp_rank = layer_idx
            train_rank = op.send_rank - 2  # 减去推理world_size
            actual_pp_rank = train_rank // tp_size

            assert actual_pp_rank == expected_pp_rank, (
                f"Layer {layer_idx} should come from PP rank {expected_pp_rank}, "
                f"but got PP rank {actual_pp_rank}"
            )

    def test_pp_mismatch_all_shards_covered(self):
        """
        验证PP不匹配时，推理端所有分片都被覆盖

        这是一个完整性检查：确保没有遗漏任何推理分片
        """
        builder = TransferPlanBuilder(
            infer_world_size=2,
            train_world_size=8,
            num_infer_engines=1,
        )

        hidden_size = 256
        tp_size = 2
        shard_size = hidden_size // tp_size

        # 简化: 只测试一层
        train_shards = []
        for tp_rank in range(tp_size):
            global_rank = 0 * tp_size + tp_rank  # PP=0
            shard = create_shard_meta(
                name="layers.0.mlp.weight",
                shape=(shard_size, hidden_size),
                global_offset=(tp_rank * shard_size, 0),
                global_rank=global_rank,
                tp_rank=tp_rank,
                pp_rank=0,
                sharding_type=ShardingType.TP_SHARDING,
                sharding_dim=0,
                num_shards=tp_size,
            )
            train_shards.append(shard)

        train_meta = create_parameter_meta(
            name="layers.0.mlp.weight",
            shards=train_shards,
            global_shape=(hidden_size, hidden_size),
        )

        infer_shards = []
        for tp_rank in range(tp_size):
            shard = create_shard_meta(
                name="layers.0.mlp.weight",
                shape=(shard_size, hidden_size),
                global_offset=(tp_rank * shard_size, 0),
                global_rank=tp_rank,
                tp_rank=tp_rank,
                pp_rank=0,
                sharding_type=ShardingType.TP_SHARDING,
                sharding_dim=0,
                num_shards=tp_size,
            )
            infer_shards.append(shard)

        infer_meta = create_parameter_meta(
            name="layers.0.mlp.weight",
            shards=infer_shards,
            global_shape=(hidden_size, hidden_size),
        )

        operations = builder.build_weights_mapping_operations(
            [infer_meta], [train_meta]
        )

        # 收集所有被覆盖的推理分片
        covered_recv_ranks = set()
        for op in operations:
            covered_recv_ranks.add(op.recv_rank)

        # 验证所有推理rank都被覆盖
        expected_ranks = set(range(tp_size))
        assert covered_recv_ranks == expected_ranks, (
            f"Not all inference ranks covered. "
            f"Expected {expected_ranks}, got {covered_recv_ranks}"
        )


# ==============================================================================
# Test Case 2: EP Mismatch (EP=32 → EP=64)
# ==============================================================================


class TestEPMismatch:
    """
    测试 Expert Parallel 不匹配场景

    训练端: EP=32, 每个EP rank有8个专家 (256 / 32 = 8)
    推理端: EP=64, 每个EP rank有4个专家 (256 / 64 = 4)

    关键: 每个推理EP rank需要从对应的训练EP rank获取一半专家
    """

    def test_ep_split_basic(self):
        """
        测试EP切分的基本场景

        简化配置:
        - 8个专家
        - 训练: EP=2, 每个EP rank有4个专家
        - 推理: EP=4, 每个EP rank有2个专家

        期望: 推理EP rank 0,1从训练EP rank 0获取; 推理EP rank 2,3从训练EP rank 1获取
        """
        # 训练 world_size = EP * TP = 2 * 1 = 2
        # 推理 world_size = EP * TP = 4 * 1 = 4
        builder = TransferPlanBuilder(
            infer_world_size=4,
            train_world_size=2,
            num_infer_engines=1,
        )

        num_experts = 8
        expert_hidden = 64
        train_ep_size = 2
        infer_ep_size = 4

        experts_per_train_ep = num_experts // train_ep_size  # 4
        experts_per_infer_ep = num_experts // infer_ep_size  # 2

        # 构建训练端元数据
        train_shards = []
        for ep_rank in range(train_ep_size):
            # 每个EP rank的专家索引范围
            expert_start = ep_rank * experts_per_train_ep
            shard = create_shard_meta(
                name="experts.weight",
                shape=(experts_per_train_ep, expert_hidden),  # 4个专家
                global_offset=(expert_start, 0),
                global_rank=ep_rank,
                tp_rank=0,
                ep_rank=ep_rank,
                sharding_type=ShardingType.EP_SHARDING,
                sharding_dim=0,
                num_shards=train_ep_size,
            )
            train_shards.append(shard)

        train_meta = create_parameter_meta(
            name="experts.weight",
            shards=train_shards,
            global_shape=(num_experts, expert_hidden),
        )
        # 手动设置replica（EP分片每个都是独立的）
        train_meta.replicas = [ParameterReplicaMeta(shards=train_shards)]

        # 构建推理端元数据
        infer_shards = []
        for ep_rank in range(infer_ep_size):
            expert_start = ep_rank * experts_per_infer_ep
            shard = create_shard_meta(
                name="experts.weight",
                shape=(experts_per_infer_ep, expert_hidden),  # 2个专家
                global_offset=(expert_start, 0),
                global_rank=ep_rank,
                tp_rank=0,
                ep_rank=ep_rank,
                sharding_type=ShardingType.EP_SHARDING,
                sharding_dim=0,
                num_shards=infer_ep_size,
            )
            infer_shards.append(shard)

        infer_meta = create_parameter_meta(
            name="experts.weight",
            shards=infer_shards,
            global_shape=(num_experts, expert_hidden),
        )
        infer_meta.replicas = [ParameterReplicaMeta(shards=infer_shards)]

        operations = builder.build_weights_mapping_operations(
            [infer_meta], [train_meta]
        )

        # 验证: 应该有4个操作，每个推理EP rank一个
        assert len(operations) == 4, f"Expected 4 operations, got {len(operations)}"

        # 验证: 推理EP 0,1应该从训练EP 0获取
        # 验证: 推理EP 2,3应该从训练EP 1获取
        for op in operations:
            recv_rank = op.recv_rank
            send_rank = op.send_rank - 4  # 减去推理world_size

            expected_train_ep = recv_rank // 2  # 0,1->0, 2,3->1
            assert send_rank == expected_train_ep, (
                f"Inference EP rank {recv_rank} should receive from "
                f"training EP rank {expected_train_ep}, but got {send_rank}"
            )

            # 验证overlap_shape正确
            assert op.overlap_shape == (experts_per_infer_ep, expert_hidden), (
                f"Expected overlap shape {(experts_per_infer_ep, expert_hidden)}, "
                f"got {op.overlap_shape}"
            )


# ==============================================================================
# Test Case 3: CP Replica Handling
# ==============================================================================


class TestCPReplicaHandling:
    """
    测试 Context Parallel 副本处理

    训练端: CP=4, 每个CP rank持有相同的attention权重
    推理端: CP=1 (无CP)

    关键: TransferPlan应该识别CP副本，可以从任意CP rank获取
    """

    def test_cp_replica_detection(self):
        """
        测试CP副本被正确识别

        场景:
        - 4个CP rank持有相同的attention权重
        - 推理端需要这个权重

        期望: TransferPlan可以从任意CP rank获取，实现负载均衡
        """
        # 简化: TP=1, CP=4
        # 训练 world_size = 4
        # 推理 world_size = 1
        builder = TransferPlanBuilder(
            infer_world_size=1,
            train_world_size=4,
            num_infer_engines=1,
        )

        hidden_size = 256

        # 训练端: 4个CP rank，每个都有完整的权重（副本）
        train_shards = []
        for cp_rank in range(4):
            shard = create_shard_meta(
                name="attn.qkv.weight",
                shape=(hidden_size, hidden_size),
                global_offset=(0, 0),
                global_rank=cp_rank,
                tp_rank=0,
                pp_rank=0,
                sharding_type=ShardingType.NO_SHARDING,
                sharding_dim=0,
                num_shards=1,
            )
            train_shards.append(shard)

        train_meta = create_parameter_meta(
            name="attn.qkv.weight",
            shards=train_shards,
            global_shape=(hidden_size, hidden_size),
        )
        # 4个副本，每个副本1个分片
        train_meta.replicas = [
            ParameterReplicaMeta(shards=[shard]) for shard in train_shards
        ]

        # 推理端: 1个rank需要完整权重
        infer_shard = create_shard_meta(
            name="attn.qkv.weight",
            shape=(hidden_size, hidden_size),
            global_offset=(0, 0),
            global_rank=0,
            tp_rank=0,
            pp_rank=0,
            sharding_type=ShardingType.NO_SHARDING,
            sharding_dim=0,
            num_shards=1,
        )
        infer_meta = create_parameter_meta(
            name="attn.qkv.weight",
            shards=[infer_shard],
            global_shape=(hidden_size, hidden_size),
        )

        operations = builder.build_weights_mapping_operations(
            [infer_meta], [train_meta]
        )

        # 验证: 应该只有1个操作（从某个CP rank获取）
        assert len(operations) == 1, f"Expected 1 operation, got {len(operations)}"

        # 验证: send_rank应该在训练rank范围内
        send_rank = operations[0].send_rank
        assert 1 <= send_rank <= 4, (
            f"send_rank should be in training range [1,4], got {send_rank}"
        )


# ==============================================================================
# Test Case 4: Data Integrity Verification
# ==============================================================================


class TestDataIntegrity:
    """
    测试数据完整性 - 模拟实际传输验证数据正确性

    这类测试验证：
    1. slice_tensor 正确切分源张量
    2. 切分后的数据可以正确拼接成目标张量
    3. 数值完全一致
    """

    def test_tp_sharding_data_transfer(self):
        """
        测试 TP 切分场景下的数据传输

        场景:
        - 训练: TP=2, 全局权重 (256, 256) 被切分为 2 个 (128, 256)
        - 推理: TP=2, 同样切分
        - 期望: 传输后推理端重建的数据与训练端完全一致
        """
        builder = TransferPlanBuilder(
            infer_world_size=2,
            train_world_size=2,
            num_infer_engines=1,
        )

        hidden_size = 256
        tp_size = 2
        shard_size = hidden_size // tp_size

        # 创建训练端的"真实"张量 (模拟)
        # global tensor: (256, 256), 用于验证
        global_tensor = torch.randn(hidden_size, hidden_size, dtype=torch.float32)

        # 训练端分片
        train_shards = []
        train_tensors = {}  # rank -> tensor
        for tp_rank in range(tp_size):
            shard = create_shard_meta(
                name="mlp.weight",
                shape=(shard_size, hidden_size),
                global_offset=(tp_rank * shard_size, 0),
                global_rank=tp_rank,
                tp_rank=tp_rank,
                pp_rank=0,
                sharding_type=ShardingType.TP_SHARDING,
                sharding_dim=0,
                num_shards=tp_size,
            )
            train_shards.append(shard)
            # 从全局张量中切出这个分片
            train_tensors[tp_rank] = global_tensor[
                tp_rank * shard_size : (tp_rank + 1) * shard_size, :
            ].clone()

        train_meta = create_parameter_meta(
            name="mlp.weight",
            shards=train_shards,
            global_shape=(hidden_size, hidden_size),
        )

        # 推理端分片（相同切分方式）
        infer_shards = []
        for tp_rank in range(tp_size):
            shard = create_shard_meta(
                name="mlp.weight",
                shape=(shard_size, hidden_size),
                global_offset=(tp_rank * shard_size, 0),
                global_rank=tp_rank,
                tp_rank=tp_rank,
                pp_rank=0,
                sharding_type=ShardingType.TP_SHARDING,
                sharding_dim=0,
                num_shards=tp_size,
            )
            infer_shards.append(shard)

        infer_meta = create_parameter_meta(
            name="mlp.weight",
            shards=infer_shards,
            global_shape=(hidden_size, hidden_size),
        )

        # 构建传输计划
        operations = builder.build_weights_mapping_operations(
            [infer_meta], [train_meta]
        )

        # 模拟传输：对每个操作，切分源张量并验证
        infer_tensors = {}  # rank -> tensor (模拟推理端接收)
        for op in operations:
            # 训练端 rank (减去推理 world_size)
            train_rank = op.send_rank - 2
            source_tensor = train_tensors[train_rank]

            # 使用 slice_tensor 切分
            sliced = slice_tensor(source_tensor, op, is_train=True)

            # 推理端接收
            recv_rank = op.recv_rank
            if recv_rank not in infer_tensors:
                infer_tensors[recv_rank] = torch.zeros(
                    shard_size, hidden_size, dtype=torch.float32
                )

            # 将切分的数据放入目标位置
            infer_tensors[recv_rank][op.inf_slices] = sliced

        # 验证：推理端重建的数据应该与原始全局张量一致
        reconstructed = torch.zeros_like(global_tensor)
        for tp_rank in range(tp_size):
            reconstructed[tp_rank * shard_size : (tp_rank + 1) * shard_size, :] = (
                infer_tensors[tp_rank]
            )

        assert torch.allclose(reconstructed, global_tensor), (
            "Reconstructed tensor does not match original!"
        )

    def test_ep_split_data_transfer(self):
        """
        测试 EP 切分场景下的数据传输 (EP翻倍)

        场景:
        - 8个专家，每个 (64,) 向量
        - 训练: EP=2, 每个EP rank有4个专家 -> 形状 (4, 64)
        - 推理: EP=4, 每个EP rank有2个专家 -> 形状 (2, 64)
        - 验证: 传输后数据值正确
        """
        builder = TransferPlanBuilder(
            infer_world_size=4,
            train_world_size=2,
            num_infer_engines=1,
        )

        num_experts = 8
        expert_hidden = 64
        train_ep_size = 2
        infer_ep_size = 4
        experts_per_train = num_experts // train_ep_size  # 4
        experts_per_infer = num_experts // infer_ep_size  # 2

        # 全局专家权重 (8, 64)
        global_experts = torch.randn(num_experts, expert_hidden, dtype=torch.float32)

        # 训练端分片
        train_shards = []
        train_tensors = {}
        for ep_rank in range(train_ep_size):
            expert_start = ep_rank * experts_per_train
            shard = create_shard_meta(
                name="experts.weight",
                shape=(experts_per_train, expert_hidden),
                global_offset=(expert_start, 0),
                global_rank=ep_rank,
                ep_rank=ep_rank,
                sharding_type=ShardingType.EP_SHARDING,
                sharding_dim=0,
                num_shards=train_ep_size,
            )
            train_shards.append(shard)
            train_tensors[ep_rank] = global_experts[
                expert_start : expert_start + experts_per_train, :
            ].clone()

        train_meta = create_parameter_meta(
            name="experts.weight",
            shards=train_shards,
            global_shape=(num_experts, expert_hidden),
        )
        train_meta.replicas = [ParameterReplicaMeta(shards=train_shards)]

        # 推理端分片
        infer_shards = []
        for ep_rank in range(infer_ep_size):
            expert_start = ep_rank * experts_per_infer
            shard = create_shard_meta(
                name="experts.weight",
                shape=(experts_per_infer, expert_hidden),
                global_offset=(expert_start, 0),
                global_rank=ep_rank,
                ep_rank=ep_rank,
                sharding_type=ShardingType.EP_SHARDING,
                sharding_dim=0,
                num_shards=infer_ep_size,
            )
            infer_shards.append(shard)

        infer_meta = create_parameter_meta(
            name="experts.weight",
            shards=infer_shards,
            global_shape=(num_experts, expert_hidden),
        )
        infer_meta.replicas = [ParameterReplicaMeta(shards=infer_shards)]

        # 构建传输计划
        operations = builder.build_weights_mapping_operations(
            [infer_meta], [train_meta]
        )

        # 模拟传输
        infer_tensors = {}
        for op in operations:
            train_rank = op.send_rank - 4  # 减去推理 world_size
            source_tensor = train_tensors[train_rank]
            sliced = slice_tensor(source_tensor, op, is_train=True)

            recv_rank = op.recv_rank
            if recv_rank not in infer_tensors:
                infer_tensors[recv_rank] = torch.zeros(
                    experts_per_infer, expert_hidden, dtype=torch.float32
                )
            infer_tensors[recv_rank][op.inf_slices] = sliced

        # 验证：重建全局专家
        reconstructed = torch.zeros_like(global_experts)
        for ep_rank in range(infer_ep_size):
            expert_start = ep_rank * experts_per_infer
            reconstructed[expert_start : expert_start + experts_per_infer, :] = (
                infer_tensors[ep_rank]
            )

        assert torch.allclose(reconstructed, global_experts), (
            "Reconstructed expert weights do not match original!"
        )


# ==============================================================================
# Test Case 5: Multi-Engine (num_infer_engines=2)
# ==============================================================================


class TestMultiEngine:
    """
    测试多推理引擎场景

    DeepSeek-V3 实际配置:
    - 推理: 2个实例，每个32卡 (总64卡)
    - 每个实例都需要完整的模型权重
    """

    def test_dual_engine_weight_distribution(self):
        """
        测试双引擎权重分发

        场景:
        - 训练: 4卡 (TP=2, PP=2)
        - 推理: 4卡 (2个引擎，每个2卡，TP=2)
        - 每个引擎都需要完整模型
        """
        builder = TransferPlanBuilder(
            infer_world_size=4,   # 总共4个推理rank
            train_world_size=4,  # 4个训练rank
            num_infer_engines=2,  # 2个引擎
        )
        # infer_instance_world_size = 4 / 2 = 2

        hidden_size = 128
        tp_size = 2
        shard_size = hidden_size // tp_size

        # 训练端: 只有PP=0的层 (简化)
        train_shards = []
        for tp_rank in range(tp_size):
            shard = create_shard_meta(
                name="layer.weight",
                shape=(shard_size, hidden_size),
                global_offset=(tp_rank * shard_size, 0),
                global_rank=tp_rank,  # PP=0的rank
                tp_rank=tp_rank,
                pp_rank=0,
                sharding_type=ShardingType.TP_SHARDING,
                sharding_dim=0,
                num_shards=tp_size,
            )
            train_shards.append(shard)

        train_meta = create_parameter_meta(
            name="layer.weight",
            shards=train_shards,
            global_shape=(hidden_size, hidden_size),
        )

        # 推理端: 2个引擎，每个引擎TP=2
        infer_shards = []
        for tp_rank in range(tp_size):
            # 只创建引擎0的分片，引擎1会通过num_infer_engines自动处理
            shard = create_shard_meta(
                name="layer.weight",
                shape=(shard_size, hidden_size),
                global_offset=(tp_rank * shard_size, 0),
                global_rank=tp_rank,  # 引擎0的rank 0,1
                tp_rank=tp_rank,
                pp_rank=0,
                sharding_type=ShardingType.TP_SHARDING,
                sharding_dim=0,
                num_shards=tp_size,
            )
            infer_shards.append(shard)

        infer_meta = create_parameter_meta(
            name="layer.weight",
            shards=infer_shards,
            global_shape=(hidden_size, hidden_size),
        )

        operations = builder.build_weights_mapping_operations(
            [infer_meta], [train_meta]
        )

        # 验证: 应该有 4 个操作
        # 引擎0: rank 0,1 各1个 = 2
        # 引擎1: rank 2,3 各1个 = 2
        assert len(operations) == 4, f"Expected 4 operations, got {len(operations)}"

        # 收集每个引擎的接收rank
        engine0_ranks = set()
        engine1_ranks = set()
        for op in operations:
            recv_rank = op.recv_rank
            if recv_rank < 2:  # 引擎0的rank
                engine0_ranks.add(recv_rank)
            else:  # 引擎1的rank
                engine1_ranks.add(recv_rank)

        # 验证两个引擎都覆盖了所有TP rank
        assert engine0_ranks == {0, 1}, f"Engine 0 should cover ranks 0,1, got {engine0_ranks}"
        assert engine1_ranks == {2, 3}, f"Engine 1 should cover ranks 2,3, got {engine1_ranks}"


# ==============================================================================
# Run Tests
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
