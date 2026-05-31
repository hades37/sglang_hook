"""Hooks for ``sglang.srt.model_executor.model_runner``.

The ``init_torch_distributed`` → ``init_distributed_environment`` +
``initialize_model_parallel`` call chain is already replaced by
``hooks/parallel_state.py``.

This module is a hook point for:

* Skipping or replacing ``load_model`` (e.g. for synthetic / dummy weights).
* Overriding ``MemoryPoolConfig`` or KV cache allocation.
* Adjusting ``forward_batch_generation`` for profiling.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply() -> None:
    """No-op — distributed init already handled by parallel_state hooks."""
    logger.info("Fake-backend model_runner hooks: no additional hooks needed")
