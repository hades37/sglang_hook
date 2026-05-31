from __future__ import annotations

import logging
import math
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed

logger = logging.getLogger(__name__)


class FakeGroupCoordinator:
    """Replaces parallel_state.GroupCoordinator for single-process fake backend.

    All ``world_size`` and topology attributes report the *logical* parallelism
    configuration so that model-weight sharding and scheduling logic see the
    intended degree of parallelism (tp_size, pp_size, dp_size, ep_size, …).
    Every collective is a **no-op**, i.e. it returns the input unchanged or
    performs a simple local split/concat that mimics a single-rank operation.
    """

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        world_size_override: Optional[int] = None,
        group_name: str = "fake",
    ):
        if world_size_override is not None:
            self._world_size = world_size_override
        else:
            self._world_size = max(len(r) for r in group_ranks)

        self.local_rank = local_rank
        self.rank = 0
        self.ranks = list(range(self._world_size))
        self.rank_in_group = 0

        self.device_group = None
        self.cpu_group = None

        self._unique_name = f"fake_{group_name}_{id(self)}"
        self._register_self()

        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.device_module = torch.get_device_module(self.device)

        self.use_pynccl = False
        self.use_pymscclpp = False
        self.use_custom_allreduce = False
        self.use_torch_symm_mem_all_reduce = False
        self.use_hpu_communicator = False
        self.use_xpu_communicator = False
        self.use_npu_communicator = False
        self.use_message_queue_broadcaster = False

        self.pynccl_comm = None
        self.pymscclpp_comm = None
        self.ca_comm = None
        self.qr_comm = None
        self.torch_symm_mem_comm = None
        self.hpu_communicator = None
        self.xpu_communicator = None
        self.npu_communicator = None
        self.mq_broadcaster = None

        self.is_symmetric_memory_enabled = False
        self.use_symmetric_memory = False
        self.is_allocation_symmetric = lambda: False
        self.debug_check_symmetric_mempool = lambda *args, **kwargs: None

        self.local_size = 1

    def _register_self(self):
        from sglang.srt.distributed.parallel_state import _groups, _get_unique_name

        unique = _get_unique_name(self._unique_name)
        self.unique_name = unique
        import weakref

        _groups[unique] = weakref.ref(self)

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def first_rank(self):
        return self.ranks[0]

    @property
    def last_rank(self):
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        return True

    @property
    def is_last_rank(self):
        return self._world_size == 1

    @property
    def next_rank(self):
        return self.ranks[0]

    @property
    def prev_rank(self):
        return self.ranks[0]

    def __repr__(self):
        return (
            f"FakeGroupCoordinator(world_size={self._world_size}, "
            f"rank_in_group={self.rank_in_group}, unique_name={self.unique_name})"
        )

    @contextmanager
    def graph_capture(self, graph_capture_context=None, stream=None):
        if stream is None:
            stream = self.device_module.Stream()
        with self.device_module.stream(stream):
            yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return input_

    def quant_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return input_

    def fused_allreduce_rmsnorm(
        self,
        input_: torch.Tensor,
        residual_inp_: torch.Tensor,
        weight_: torch.Tensor,
        eps: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return None

    def _all_reduce_out_place(
        self, input_: torch.Tensor, outplace_all_reduce_method: str
    ) -> torch.Tensor:
        return input_

    def _all_reduce_in_place(self, input_: torch.Tensor) -> None:
        pass

    def _reduce_scatter_tensor(
        self, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        wsize = self._world_size
        if wsize == 1:
            output.copy_(input)
            return output
        split_size = input.shape[0] // wsize
        output.copy_(input[:split_size])
        return output

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor):
        return self._reduce_scatter_tensor(output, input)

    def reduce_scatter(
        self, output: torch.Tensor, input_list: List[torch.Tensor]
    ) -> None:
        output.copy_(input_list[0])
        return output

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        wsize = self._world_size
        if sizes is not None:
            chunk_size = sizes[0]
        else:
            chunk_size = input_.shape[0] // wsize
        output_shape = (chunk_size,) + input_.shape[1:]
        if output is None:
            output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)
        output.copy_(input_[:chunk_size])
        return output

    def _all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        wsize = self._world_size
        if wsize == 1:
            output.copy_(input)
            return
        chunk_size = input.shape[0]
        for i in range(wsize):
            output[i * chunk_size : (i + 1) * chunk_size].copy_(input)

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        self._all_gather_into_tensor(output, input)

    def cp_all_gather_into_tensor_async(
        self, output: torch.Tensor, input: torch.Tensor, stream: torch.cuda.Stream
    ):
        self.all_gather_into_tensor(output, input)

    def all_gather(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        output_tensor_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        wsize = self._world_size
        if wsize == 1:
            if output_tensor_list is not None:
                output_tensor_list[0].copy_(input_)
                return None
            return input_

        if output_tensor_list is not None:
            for idx, t in enumerate(output_tensor_list):
                t.copy_(input_)
            return None

        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = input_size[:dim] + (wsize * input_size[dim],) + input_size[dim + 1 :]
        output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
        slices_per_dim = [slice(None)] * input_.dim()
        for i in range(wsize):
            slices_per_dim[dim] = slice(i * input_size[dim], (i + 1) * input_size[dim])
            output_tensor[tuple(slices_per_dim)] = input_
        return output_tensor

    def all_gatherv(
        self,
        input_: Union[torch.Tensor, List[torch.Tensor]],
        sizes: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        return [input_] if isinstance(input_, torch.Tensor) else input_

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> Optional[torch.Tensor]:
        if self.rank_in_group == dst:
            return input_
        return None

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        return input_

    def broadcast_object(self, obj: Optional[Any] = None, src: int = 0):
        return obj

    def broadcast_object_list(
        self, obj_list: List[Any], src: int = 0, group=None
    ):
        return obj_list

    def all_gather_object(self, obj: Any) -> List[Any]:
        return [obj] * self._world_size

    def send_object(
        self, obj: Any, dst: int, async_send: bool = False
    ) -> List:
        return []

    def recv_object(self, src: int) -> Any:
        raise RuntimeError("FakeGroupCoordinator.recv_object should not be called")

    def broadcast_tensor_dict(
        self,
        tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        src: int = 0,
        group=None,
        metadata_group=None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        return tensor_dict

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None,
        all_gather_group: Optional[FakeGroupCoordinator] = None,
        async_send: bool = False,
    ):
        return tensor_dict
