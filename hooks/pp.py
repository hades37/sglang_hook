"""Hooks for Pipeline-Parallel (PP) logic.

The PP group is created as a ``FakeGroupCoordinator`` with
``world_size`` = configured ``pp_size`` and ``rank_in_group`` = 0.
Since ``pp_size`` is typically 1 (or forced to 1 by the fake backend),
``pp_group.is_last_rank`` is always ``True`` and the scheduler runs
the appropriate event loop for the last (only) pipeline stage.

No additional hooks are needed — PP send/recv operations are identity
through ``FakeGroupCoordinator``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply() -> None:
    """No-op — PP communication handled by FakeGroupCoordinator."""
    logger.info("Fake-backend PP hooks: handled by FakeGroupCoordinator")
