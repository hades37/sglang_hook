"""Hooks for ``sglang.srt.distributed.parallel_state``.

Replaces distributed-initialization with a single-process fake backend so that
the full scheduling / tokenization / inference pipeline runs on **one** GPU while
still partitioning model weights and scheduling logic as if ``tp_size``, ``pp_size``,
``dp_size``, ``ep_size`` were at their configured values.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Callable, List, Optional

import torch
import torch.distributed

from sglang_hook.utils.fake_group import FakeGroupCoordinator

logger = logging.getLogger(__name__)

_original_init_distributed_environment: Optional[Callable] = None
_original_init_model_parallel_group: Optional[Callable] = None
_original_initialize_model_parallel: Optional[Callable] = None
_original_torch_get_world_size: Optional[Callable] = None
_original_torch_get_rank: Optional[Callable] = None


def _get_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def apply() -> None:
    """Inject fake-backend hooks into ``sglang.srt.distributed.parallel_state``."""
    import sglang.srt.distributed.parallel_state as ps

    global _original_init_distributed_environment, _original_init_model_parallel_group
    global _original_initialize_model_parallel, _original_torch_get_world_size
    global _original_torch_get_rank

    _original_init_distributed_environment = ps.init_distributed_environment
    _original_init_model_parallel_group = ps.init_model_parallel_group
    _original_initialize_model_parallel = ps.initialize_model_parallel
    _original_torch_get_world_size = torch.distributed.get_world_size
    _original_torch_get_rank = torch.distributed.get_rank

    ps.init_distributed_environment = _fake_init_distributed_environment
    ps.init_world_group = _fake_init_world_group
    ps.init_model_parallel_group = _fake_init_model_parallel_group
    ps.initialize_model_parallel = _fake_initialize_model_parallel

    logger.info("Fake-backend parallel_state hooks applied")


def _fake_init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: Optional[int] = None,
    moe_a2a_backend: Optional[str] = None,
    recovered_rank: bool = False,
):
    """Init torch.distributed with a single process (gloo backend, world_size=1).

    The *caller's* ``world_size`` / ``rank`` are saved so that later the
    fake ``initialize_model_parallel`` can compute groups using the
    intended parallelism degrees.
    """
    import sglang.srt.distributed.parallel_state as ps

    if torch.distributed.is_initialized():
        return

    if distributed_init_method == "env://":
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_get_free_port()))
        distributed_init_method = "env://"

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=distributed_init_method,
        world_size=1,
        rank=0,
        timeout=None,
    )

    if local_rank == -1:
        local_rank = 0

    ps._WORLD = FakeGroupCoordinator(
        group_ranks=[list(range(max(world_size, 1)))],
        local_rank=local_rank,
        world_size_override=max(world_size, 1),
        group_name="world",
    )

    logger.debug(f"Fake distributed environment ready (logical world_size={world_size})")


def _fake_init_world_group(
    ranks: List[int],
    local_rank: int,
    backend: str,
    recovered_rank: bool = False,
) -> FakeGroupCoordinator:
    return FakeGroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        group_name="world",
    )


def _fake_init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_pynccl: Optional[bool] = None,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
    use_mscclpp_allreduce: Optional[bool] = None,
    use_torch_symm_mem_allreduce: Optional[bool] = None,
    recovered_rank: bool = False,
) -> FakeGroupCoordinator:
    world_size = max(len(r) for r in group_ranks) if group_ranks else 1
    return FakeGroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        world_size_override=world_size,
        group_name=group_name or "model_parallel",
    )


def _fake_initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    attention_data_parallel_size: int = 1,
    attention_context_model_parallel_size: int = 1,
    moe_data_model_parallel_size: int = 1,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
    enable_symm_mem: bool = False,
    recovered_rank: bool = False,
) -> None:
    """Fake implementation that creates single-rank groups for every parallelism dimension.

    The original ``initialize_model_parallel`` depends on ``torch.distributed.get_world_size()``
    returning the *real* process count.  Since we only have one process, this function
    uses the *requested* sizes directly.
    """
    import sglang.srt.distributed.parallel_state as ps

    world_size = tensor_model_parallel_size * pipeline_model_parallel_size

    # ---- TP group ----
    assert ps._TP is None, "tensor model parallel group is already initialized"
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    group_ranks = [
        list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
        for i in range(num_tensor_model_parallel_groups)
    ]
    ps._TP = _fake_init_model_parallel_group(
        group_ranks,
        ps._WORLD.local_rank,
        backend or "gloo",
        use_message_queue_broadcaster=os.environ.get("SGLANG_USE_MESSAGE_QUEUE_BROADCASTER") == "1",
        group_name="tp",
        recovered_rank=recovered_rank,
    )

    # ---- duplicate TP for PD-Multiplexing (if requested) ----
    if duplicate_tp_group:
        assert ps._PDMUX_PREFILL_TP_GROUP is None
        ps._PDMUX_PREFILL_TP_GROUP = _fake_init_model_parallel_group(
            group_ranks,
            ps._WORLD.local_rank,
            backend or "gloo",
            use_message_queue_broadcaster=os.environ.get("SGLANG_USE_MESSAGE_QUEUE_BROADCASTER") == "1",
            group_name="pdmux_prefill_tp",
            recovered_rank=recovered_rank,
        )

    # ---- ATTN_TP / ATTN_CP ----
    attn_dp_size = attention_data_parallel_size
    attn_cp_size = attention_context_model_parallel_size
    attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size

    if attn_cp_size == tensor_model_parallel_size:
        ps._ATTN_CP = ps._TP
    else:
        group_ranks = []
        for tp_gi in range(num_tensor_model_parallel_groups):
            for dp_idx in range(attn_dp_size):
                for attn_tp_idx in range(attn_tp_size):
                    st = tp_gi * tensor_model_parallel_size + dp_idx * attn_tp_size * attn_cp_size + attn_tp_idx
                    en = tp_gi * tensor_model_parallel_size + (dp_idx + 1) * attn_tp_size * attn_cp_size + attn_tp_idx
                    ranks = list(range(st, en, attn_tp_size))
                    group_ranks.append(ranks)
        ps._ATTN_CP = _fake_init_model_parallel_group(
            group_ranks,
            ps._WORLD.local_rank,
            backend or "gloo",
            use_message_queue_broadcaster=os.environ.get("SGLANG_USE_MESSAGE_QUEUE_BROADCASTER") == "1",
            group_name="attn_cp",
            recovered_rank=recovered_rank,
        )

    if attn_tp_size == tensor_model_parallel_size:
        ps._ATTN_TP = ps._TP
    else:
        group_ranks = []
        for tp_gi in range(num_tensor_model_parallel_groups):
            for cp_dp_combined_idx in range(attn_cp_size * attn_dp_size):
                st = tp_gi * tensor_model_parallel_size + cp_dp_combined_idx * attn_tp_size
                en = tp_gi * tensor_model_parallel_size + (cp_dp_combined_idx + 1) * attn_tp_size
                ranks = list(range(st, en))
                group_ranks.append(ranks)
        ps._ATTN_TP = _fake_init_model_parallel_group(
            group_ranks,
            ps._WORLD.local_rank,
            backend or "gloo",
            use_pynccl=False,
            use_custom_allreduce=False,
            use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False,
            use_message_queue_broadcaster=os.environ.get("SGLANG_USE_MESSAGE_QUEUE_BROADCASTER") == "1",
            group_name="attention_tp",
            recovered_rank=recovered_rank,
        )

    # ---- MOE_DP / MOE_EP / MOE_TP ----
    moe_ep_size = expert_model_parallel_size
    moe_dp_size = moe_data_model_parallel_size
    moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size

    if attn_cp_size > moe_dp_size:
        ps._MOE_DP = ps._ATTN_CP
    elif moe_dp_size == tensor_model_parallel_size:
        ps._MOE_DP = ps._TP
    else:
        group_ranks = []
        for tp_gi in range(num_tensor_model_parallel_groups):
            for tp_ep_combined_idx in range(moe_tp_size * moe_ep_size):
                st = tp_gi * tensor_model_parallel_size + tp_ep_combined_idx
                en = (tp_gi + 1) * tensor_model_parallel_size + tp_ep_combined_idx
                ranks = list(range(st, en, moe_tp_size * moe_ep_size))
                group_ranks.append(ranks)
        ps._MOE_DP = _fake_init_model_parallel_group(
            group_ranks,
            ps._WORLD.local_rank,
            backend or "gloo",
            group_name="moe_dp",
            recovered_rank=recovered_rank,
        )

    if moe_ep_size == tensor_model_parallel_size:
        ps._MOE_EP = ps._TP
    else:
        group_ranks = []
        for tp_gi in range(num_tensor_model_parallel_groups):
            for moe_dp_idx in range(moe_dp_size):
                for moe_tp_idx in range(moe_tp_size):
                    st = tp_gi * tensor_model_parallel_size + moe_dp_idx * moe_ep_size * moe_tp_size + moe_tp_idx
                    en = st + moe_ep_size * moe_tp_size
                    ranks = list(range(st, en, moe_tp_size))
                    group_ranks.append(ranks)
        ps._MOE_EP = _fake_init_model_parallel_group(
            group_ranks,
            ps._WORLD.local_rank,
            backend or "gloo",
            use_pynccl=False,
            use_custom_allreduce=False,
            group_name="moe_ep",
            recovered_rank=recovered_rank,
        )

    if moe_tp_size == tensor_model_parallel_size:
        ps._MOE_TP = ps._TP
    else:
        group_ranks = []
        for tp_gi in range(num_tensor_model_parallel_groups):
            for ep_dp_combined_idx in range(moe_ep_size * moe_dp_size):
                st = tp_gi * tensor_model_parallel_size + ep_dp_combined_idx * moe_tp_size
                en = tp_gi * tensor_model_parallel_size + (ep_dp_combined_idx + 1) * moe_tp_size
                ranks = list(range(st, en))
                group_ranks.append(ranks)
        ps._MOE_TP = _fake_init_model_parallel_group(
            group_ranks,
            ps._WORLD.local_rank,
            backend or "gloo",
            use_pynccl=False,
            use_custom_allreduce=False,
            group_name="moe_tp",
            recovered_rank=recovered_rank,
        )

    # ---- PP group ----
    assert ps._PP is None, "pipeline model parallel group is already initialized"
    num_pp_groups = world_size // pipeline_model_parallel_size
    group_ranks = [
        list(range(i, world_size, num_pp_groups)) for i in range(num_pp_groups)
    ]
    ps._PP = _fake_init_model_parallel_group(
        group_ranks,
        ps._WORLD.local_rank,
        backend or "gloo",
        use_custom_allreduce=False,
        group_name="pp",
        recovered_rank=recovered_rank,
    )
