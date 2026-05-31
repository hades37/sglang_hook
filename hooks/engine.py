"""Hooks for ``sglang.srt.entrypoints.engine``.

Two hooks are installed:

1. ``Engine._launch_subprocesses`` — forces ``dp_size=1, nnodes=1`` *before*
   ``PortArgs`` and ``TokenizerManager`` are initialized.

2. ``Engine._launch_scheduler_processes`` — spawns only **one** scheduler
   subprocess (tp_rank=0, pp_rank=0), regardless of configured ``tp_size``.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import sglang.srt.entrypoints.engine as engine_mod
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)

_original_launch_subprocesses: Optional[Callable] = None
_original_launch_scheduler_processes: Optional[Callable] = None


def apply() -> None:
    global _original_launch_subprocesses, _original_launch_scheduler_processes

    _original_launch_subprocesses = engine_mod.Engine._launch_subprocesses
    engine_mod.Engine._launch_subprocesses = _fake_launch_subprocesses

    _original_launch_scheduler_processes = (
        engine_mod.Engine._launch_scheduler_processes
    )
    engine_mod.Engine._launch_scheduler_processes = (
        _fake_launch_scheduler_processes
    )

    logger.info("Fake-backend engine hooks applied")


def _fake_launch_subprocesses(
    cls,
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable,
    run_scheduler_process_func: Callable,
    run_detokenizer_process_func: Callable,
    port_args: Optional[PortArgs] = None,
) -> Tuple:
    """Force dp_size=1, nnodes=1 before the real launch flow runs.

    This ensures ``PortArgs.init_new`` uses IPC (ZMQ) ports and
    ``FanOutCommunicator`` uses ``fan_out=1``.
    """
    saved = server_args.dp_size, server_args.nnodes

    server_args.dp_size = 1
    server_args.nnodes = 1

    try:
        return _original_launch_subprocesses(
            cls,
            server_args=server_args,
            init_tokenizer_manager_func=init_tokenizer_manager_func,
            run_scheduler_process_func=run_scheduler_process_func,
            run_detokenizer_process_func=run_detokenizer_process_func,
            port_args=port_args,
        )
    finally:
        server_args.dp_size, server_args.nnodes = saved


def _fake_launch_scheduler_processes(
    cls,
    server_args: ServerArgs,
    port_args: PortArgs,
    run_scheduler_process_func: Callable,
) -> Tuple:
    """Spawn exactly **one** scheduler subprocess (tp_rank=0, pp_rank=0)."""
    import multiprocessing as mp

    from sglang.srt.entrypoints.engine import (
        SchedulerInitResult,
        _compute_parallelism_ranks,
        _wait_for_scheduler_ready,
    )
    from sglang.srt.utils import maybe_reindex_device_id
    from sglang.srt.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    scheduler_procs: List[mp.Process] = []
    scheduler_pipe_readers: List = []

    reader, writer = mp.Pipe(duplex=False)
    gpu_id = server_args.base_gpu_id

    attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
        server_args, tp_rank=0
    )

    memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=server_args.enable_memory_saver
    )

    with maybe_reindex_device_id(gpu_id) as gpu_id:
        proc = mp.Process(
            target=run_scheduler_process_func,
            args=(
                server_args,
                port_args,
                gpu_id,
                0,  # tp_rank
                attn_cp_rank,
                moe_dp_rank,
                moe_ep_rank,
                0,  # pp_rank
                None,  # dp_rank
                writer,
            ),
        )
        with memory_saver_adapter.configure_subprocess():
            proc.start()

    scheduler_procs.append(proc)
    scheduler_pipe_readers.append(reader)

    all_child_pids = [proc.pid for proc in scheduler_procs]
    scheduler_infos: List = []

    def wait_for_ready():
        infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
        scheduler_infos.extend(infos)

    def wait_for_completion():
        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler {proc.pid} terminated with {proc.exitcode}"
            )

    return (
        SchedulerInitResult(
            scheduler_infos=scheduler_infos,
            all_child_pids=all_child_pids,
            wait_for_ready=wait_for_ready,
            wait_for_completion=wait_for_completion,
        ),
        scheduler_procs,
    )
