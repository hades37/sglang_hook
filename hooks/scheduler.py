"""Hooks for ``sglang.srt.managers.scheduler``.

Since we only have one scheduler process (tp_rank=0, pp_rank=0), the
per-rank IPC setup in ``init_ipc_channels`` naturally binds only the
rank-0 ZMQ sockets.  The intra-TP broadcast logic (``recv_requests``)
is triggered when ``self.tp_size != 1``, but since ``self.tp_cpu_group``
is ``None`` in the fake backend, ``broadcast_pyobj`` falls back to the
default ``gloo`` world group (size 1) → identity.

No additional hooks are currently needed, but this module exists as a
hook point for:

* Overriding ``Scheduler.event_loop`` for debugging / tracing.
* Injecting synthetic request streams.
* Adjusting ``schedule_policy`` parameters at runtime.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply() -> None:
    """No-op — scheduler naturally works with 1 process."""
    logger.info("Fake-backend scheduler hooks: no additional hooks needed")
