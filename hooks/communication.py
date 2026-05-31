"""Hooks for ``sglang.srt.distributed.communication_op`` and ``sglang.srt.layers.communicator``.

All tensor collective operations (all-reduce, reduce-scatter, all-gather, …)
are routed through ``GroupCoordinator`` methods.  Since every group is now a
``FakeGroupCoordinator`` (see ``hooks/parallel_state.py``) whose collectives
are identity / local-split / local-repeat, **no additional hooks are needed**
at the communication layer.

This module exists as a hook point for future extensions (e.g. tracing,
latency injection, or handling of edge cases with MoE / fused kernels).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply() -> None:
    """No-op — all communication is already handled by FakeGroupCoordinator."""
    logger.info("Fake-backend communication hooks: no additional hooks needed")
