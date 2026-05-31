"""Hooks for Tensor-Parallel (TP) logic.

TP weight sharding and communication are automatically handled by the
``FakeGroupCoordinator`` (TP group reports ``world_size`` = configured
``tp_size``, ``rank_in_group`` = 0).  Model layers (``ColumnParallelLinear``,
``RowParallelLinear``, ``VocabParallelEmbedding``) read
``get_tensor_model_parallel_world_size()`` and
``get_tensor_model_parallel_rank()`` to determine the shard to load — rank 0
gets the first partition, matching real TP semantics.

All TP collectives (all-reduce, reduce-scatter) are identity/no-op since
``FakeGroupCoordinator`` has no underlying ``ProcessGroup``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply() -> None:
    """No-op — TP sharding and communication handled by FakeGroupCoordinator."""
    logger.info("Fake-backend TP hooks: handled by FakeGroupCoordinator")
