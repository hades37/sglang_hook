"""Hooks for ``sglang.srt.model_executor.model_runner``.

Overrides ``ModelRunner.init_torch_distributed`` to skip GPU device
initialization when ``SGLANG_HOOK`` is active.  In the fake
backend only one process runs — GPU setup is either unnecessary (CPU fallback)
or handled by a separate real-GPU path.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _fake_init_torch_distributed(original_fn, self):
    import torch

    if not torch.cuda.is_available() and self.device == "cuda":
        logger.warning(
            "Fake backend: CUDA not available, falling back to CPU device"
        )
        self.device = "cpu"
        self.gpu_id = 0

    return original_fn(self)
