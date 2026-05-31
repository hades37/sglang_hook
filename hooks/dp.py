"""Hooks for Data-Parallel (DP) logic.

In the fake backend ``dp_size`` is forced to 1 at the engine level
(``hooks/engine.py``), which:

* Prevents the ``DataParallelController`` from being spawned.
* Makes ``FanOutCommunicator`` use ``fan_out=1`` (single scheduler).
* Skips DP-attention all-gather / reduce-scatter (since ``dp_size=1``
  the attention layers do not enter the DP code path).

If future versions need to simulate ``dp_size > 1`` with named DP ranks
mapped to a single process, additional hooks would go here (e.g.
request-to-rank routing, per-rank metrics).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply() -> None:
    """No-op — DP is forced to size 1 by the engine hook."""
    logger.info("Fake-backend DP hooks: dp_size forced to 1 by engine hook")
