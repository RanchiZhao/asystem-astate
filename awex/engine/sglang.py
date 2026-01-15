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

import time
from typing import Any, Dict, List, Optional, Union

from awex import logging
from awex.config import InferenceConfig
from awex.engine.core import InferenceEngine
from awex.reader.weights_reader import get_weights_exchange_reader
from awex.util.gpu import get_gpu_status

logger = logging.getLogger(__name__)


class SGLangEngine(InferenceEngine):
    def __init__(self, config: Union[Dict[str, Any], InferenceConfig], sgl_engine):
        super().__init__(sgl_engine.tokenizer_manager.model_config)
        if isinstance(config, dict):
            config = InferenceConfig.from_dict(config)
        self._config = config
        self._sgl_engine = sgl_engine
        self.node_rank = config.node_rank
        self.released_tags = set()
        self.weights_exchange_reader = None
        self.rank_coordinate = f"{config.engine_rank}-{self.node_rank}"
        self._initialized = False

        # In cross-node colocate mode, each engine independently handles weight updates
        # because execute_task_in_model_worker cannot broadcast across nodes.
        # When awex_per_node_mode is True, all engines (not just node_rank=0) initialize
        # WeightsReader and handle weight updates for their local 8 GPUs.
        self._awex_per_node_mode = getattr(config, 'awex_per_node_mode', False)
        if self._awex_per_node_mode:
            logger.info(
                f"[SGLangEngine] {self.rank_coordinate}: Per-node mode enabled - "
                f"this engine will independently handle weight updates"
            )

    @property
    def engine_name(self):
        return "sglang"

    @property
    def config(self):
        return self._config

    def initialize(self) -> None:
        import time
        # In per-node mode, all engines initialize (not just node_rank=0)
        # because each engine independently handles weight updates for its 8 GPUs.
        should_initialize = self._awex_per_node_mode or self.config.node_rank == 0

        if should_initialize:
            logger.info(
                f"Start to initialize weights exchange reader for {self.rank_coordinate} "
                f"(per_node_mode={self._awex_per_node_mode})"
            )
            self._initialized = True
            t0 = time.time()
            logger.info(f"[PROFILE] {self.rank_coordinate} Calling get_weights_exchange_reader...")
            self.weights_exchange_reader = get_weights_exchange_reader(self)
            t1 = time.time()
            logger.info(f"[PROFILE] {self.rank_coordinate} get_weights_exchange_reader took {t1-t0:.2f}s")
            logger.info(f"[PROFILE] {self.rank_coordinate} Calling weights_exchange_reader.initialize()...")
            self.weights_exchange_reader.initialize()
            t2 = time.time()
            logger.info(f"[PROFILE] {self.rank_coordinate} initialize() took {t2-t1:.2f}s")
            logger.info(
                f"Finished initializing weights exchange reader for {self.rank_coordinate}"
            )
        else:
            logger.info(
                f"Skip initializing weights exchange reader for {self.rank_coordinate}"
            )

    def update_weights_from_disk(
        self, model_path: str, load_format: Optional[str] = None
    ):
        """Update model weights for inference."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call setup_model() first.")
        logger.info(
            f"Start to update weights from disk for step {self.global_step} for "
            f"{self.rank_coordinate}, path: {model_path}, load_format: {load_format}"
        )
        if self.node_rank != 0:
            logger.info("Non-zero rank node, skipping update weights from disk")
            return
        self._sgl_engine.update_weights_from_disk(
            model_path=model_path, load_format=load_format
        )
        logger.info(
            f"Finished updating weights from disk for step {self.global_step} for "
            f"{self.rank_coordinate}, path: {model_path}, load_format: {load_format}"
        )

    def update_weights(self, **kwargs):
        # Use step_id from kwargs if provided, otherwise fallback to global_step
        step_id = kwargs.pop('step_id', self.global_step)

        # In per-node mode, all engines execute update_weights (not just node_rank=0)
        # because each engine independently handles weight updates for its 8 GPUs.
        # In standard mode, only node_rank=0 workers have initialized weights_exchange_reader
        # Other workers skip the update (they participate via execute_task_in_model_worker)
        should_update = self._awex_per_node_mode or self.node_rank == 0

        if not should_update:
            logger.info(
                f"Non-zero rank node {self.rank_coordinate}, skipping update_weights"
            )
            return

        logger.info(
            f"Start to update weights for step {step_id} for {self.rank_coordinate} "
            f"(per_node_mode={self._awex_per_node_mode})"
        )
        start_time = time.time()
        self.weights_exchange_reader.update_weights(step_id=step_id, **kwargs)
        duration = time.time() - start_time
        logger.info(
            f"Finished updating weights for step {step_id} for {self.rank_coordinate}, "
            f"took {duration:.3f} seconds"
        )

    def release_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        tags = tags or ["kv_cache", "weights"]
        if isinstance(tags, str):
            tags = [tags]
        # In per-node mode, all engines can release memory (not just node_rank=0)
        should_release = self._awex_per_node_mode or self.node_rank == 0
        if self._initialized and should_release:
            logger.info(
                f"Release memory occupation {tags}, released_tags {self.released_tags}"
            )
            if set(tags) - self.released_tags != set(tags):
                tags = list(set(tags) - self.released_tags)
            self.released_tags.update(tags)
            if not tags:
                logger.info("No memory occupation to release")
                return
            logger.info(f"Start to release memory occupation {tags}")
            logger.info(f"GPU status before release:\n{get_gpu_status()}")
            self._sgl_engine.release_memory_occupation(tags=tags)
            logger.info("Finished releasing memory occupation")
            logger.info(f"GPU status after release:\n{get_gpu_status()}")

    def resume_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        """Resume memory occupation for the engine.
        tags: kv_cache, weights, default is both
        """
        tags = tags or ["kv_cache", "weights"]
        if isinstance(tags, str):
            tags = [tags]
        # In per-node mode, all engines can resume memory (not just node_rank=0)
        should_resume = self._awex_per_node_mode or self.node_rank == 0
        if self._initialized and should_resume:
            logger.info(
                f"Resume memory occupation {tags}, released_tags {self.released_tags}"
            )
            tags = list(self.released_tags & set(tags))
            self.released_tags.difference_update(tags)
            if not tags:
                logger.info("No memory occupation to resume")
                return
            logger.info(f"Start to resume memory occupation {tags}")
            logger.info(f"GPU status before resume:\n{get_gpu_status()}")
            self._sgl_engine.resume_memory_occupation(tags=tags)
            logger.info("Finished resuming memory occupation")
            logger.info(f"GPU status after resume:\n{get_gpu_status()}")

    def execute_task_in_model_worker(self, fn, **kwargs):
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call `initialize` first.")
        # In per-node mode, all engines can execute tasks (not just node_rank=0)
        should_execute = self._awex_per_node_mode or self.node_rank == 0
        if not should_execute:
            raise RuntimeError(
                f"Non-zero rank node {self.rank_coordinate} is not allowed to "
                f"execute task in model workers"
            )
        return self._sgl_engine.execute_task_in_model_worker(fn, **kwargs)

    @property
    def num_engines(self):
        return self._config.num_engines

    @property
    def engine_rank(self):
        return self._config.engine_rank


def extract_sgl_config(config: Dict[str, Any]) -> Dict[str, Any]:
    from sglang.srt.server_args import ServerArgs

    engine_kwargs = {
        k: v for k, v in config.items() if k in ServerArgs.__dataclass_fields__
    }
    return engine_kwargs
