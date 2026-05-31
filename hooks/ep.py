"""Hooks for Expert-Parallel (EP) logic.

The EP, MoE-DP, and MoE-TP groups are created as ``FakeGroupCoordinator``
instances.  With ``ep_size`` or ``moe_dp_size`` > 1 the sharding logic is
preserved (rank 0 gets the first expert shard), but cross-rank MoE
communication is identity / no-op.

No additional hooks are needed unless custom MoE transfer backends
(e.g. Mooncake, DeepEP) are active — in that case their
initialization would need to be stubbed here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply() -> None:
    """No-op — EP/MoE communication handled by FakeGroupCoordinator."""
    logger.info("Fake-backend EP hooks: handled by FakeGroupCoordinator")
