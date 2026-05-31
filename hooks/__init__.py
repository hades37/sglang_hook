"""SGLang fake-backend hooks — register and apply.

Uses SGLang's native ``HookRegistry`` so that hooks are applied at the right
time (when ``load_plugins()`` calls ``HookRegistry.apply_hooks()``) in both
the main process and every forked subprocess.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from sglang_hook.hooks.engine import (
    _fake_launch_scheduler_processes,
    _fake_launch_subprocesses,
)
from sglang_hook.hooks.parallel_state import (
    _fake_init_distributed_environment,
    _fake_init_model_parallel_group,
    _fake_init_world_group,
    _fake_initialize_model_parallel,
)

logger = logging.getLogger(__name__)

_HOOKS_APPLIED = False


def apply_all() -> None:
    """Register all fake-backend hooks.

    Idempotent — safe to call multiple times.  The actual monkey-patching
    happens when ``HookRegistry.apply_hooks()`` runs inside SGLang's
    ``load_plugins()``.
    """
    global _HOOKS_APPLIED
    if _HOOKS_APPLIED:
        return
    _HOOKS_APPLIED = True

    HookRegistry.register(
        "sglang.srt.distributed.parallel_state.init_distributed_environment",
        _fake_init_distributed_environment,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.distributed.parallel_state.init_world_group",
        _fake_init_world_group,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.distributed.parallel_state.init_model_parallel_group",
        _fake_init_model_parallel_group,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.distributed.parallel_state.initialize_model_parallel",
        _fake_initialize_model_parallel,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.entrypoints.engine.Engine._launch_subprocesses",
        _fake_launch_subprocesses,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.entrypoints.engine.Engine._launch_scheduler_processes",
        _fake_launch_scheduler_processes,
        HookType.AROUND,
    )

    logger.info("Fake-backend hooks registered (6 hooks)")
