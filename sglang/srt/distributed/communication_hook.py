# sglang/srt/distributed/communication_hook.py

import os
import torch
import torch.distributed as dist
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

_HOOK_ENABLED = os.environ.get("SGLANG_HOOK", "0") == "1"
_HOOK_INITIALIZED = False
_DIST_INITIALIZED = False
_PYNCCL_HOOKED = False

_original_funcs = {}
_default_group = None


class FakeWork:
    """模拟异步操作的 Work 对象"""
    def wait(self):
        return True
    
    def is_completed(self):
        return True
    
    def get_future(self):
        fut = torch.futures.Future()
        fut.set_result(None)
        return fut


class FakeProcessGroup:
    """假的 ProcessGroup 对象"""
    def __init__(self, ranks=None, rank=0, size=1, backend="nccl"):
        self._ranks = ranks or [0]
        self._rank = rank
        self._size = size
        self._backend = backend
        self.device_group = self
    
    def rank(self):
        return self._rank
    
    def size(self):
        return self._size
    
    def backend(self):
        return self._backend
    
    def __repr__(self):
        return f"FakeProcessGroup(ranks={self._ranks}, backend={self._backend})"


class FakePyNcclCommunicator:
    """假的 PyNcclCommunicator，不做任何实际 NCCL 通信"""
    
    def __init__(self, group, device, library_path=None):
        print(f"[HOOK] FakePyNcclCommunicator.__init__: device={device}", flush=True)
        self.rank = 0
        self.world_size = 1
        self.group = group
        self.available = True
        self.disabled = True  # 关键：设置为 disabled，这样所有通信操作都会被跳过
        
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.comm = None
        self.nccl = None
        self.unique_id = None
    
    def all_reduce(self, in_tensor, op=None, stream=None):
        if self.disabled:
            return in_tensor  # 返回原 tensor
        return in_tensor
    
    def all_gather(self, output_tensor, input_tensor, stream=None):
        if self.disabled:
            # 简单复制
            chunk_size = input_tensor.numel()
            output_tensor.view(-1)[:chunk_size].copy_(input_tensor.view(-1))
            return
        return
    
    def reduce_scatter(self, output_tensor, input_tensor, op=None, stream=None):
        if self.disabled:
            chunk_size = output_tensor.numel()
            output_tensor.copy_(input_tensor.view(-1)[:chunk_size].view(output_tensor.shape))
            return
        return
    
    def send(self, tensor, dst, stream=None):
        return
    
    def recv(self, tensor, src, stream=None):
        tensor.zero_()
        return
    
    def broadcast(self, tensor, src, stream=None):
        return


def is_hook_enabled() -> bool:
    return _HOOK_ENABLED


def is_active_rank(tp_rank: int, dp_rank: int, pp_rank: int) -> bool:
    if not _HOOK_ENABLED:
        return True
    return tp_rank == 0 and dp_rank == 0


# ============== 分布式初始化 Hooks ==============

def fake_init_process_group(backend=None, init_method=None, world_size=-1, rank=-1, 
                            store=None, group_name='', pg_options=None, **kwargs):
    global _DIST_INITIALIZED, _default_group
    if _HOOK_ENABLED:
        print(f"[HOOK] fake_init_process_group: backend={backend}, world_size={world_size}, rank={rank}", flush=True)
        _default_group = FakeProcessGroup(ranks=list(range(world_size)), rank=rank, size=world_size, backend=backend or "nccl")
        _DIST_INITIALIZED = True
        return None
    return _original_funcs['init_process_group'](
        backend=backend, init_method=init_method, world_size=world_size, 
        rank=rank, store=store, group_name=group_name, pg_options=pg_options, **kwargs
    )


def fake_is_initialized():
    if _HOOK_ENABLED:
        return _DIST_INITIALIZED
    return _original_funcs['is_initialized']()


def fake_get_rank(group=None):
    if _HOOK_ENABLED:
        if group is not None and isinstance(group, FakeProcessGroup):
            return group.rank()
        return 0
    return _original_funcs['get_rank'](group)


def fake_get_world_size(group=None):
    if _HOOK_ENABLED:
        if group is not None and isinstance(group, FakeProcessGroup):
            return group.size()
        if _default_group:
            return _default_group.size()
        return 1
    return _original_funcs['get_world_size'](group)


def fake_get_backend(group=None):
    if _HOOK_ENABLED:
        if group is not None and isinstance(group, FakeProcessGroup):
            return group.backend()
        if _default_group:
            return _default_group.backend()
        return "nccl"
    return _original_funcs['get_backend'](group)


def fake_get_default_group():
    if _HOOK_ENABLED:
        return _default_group
    return _original_funcs.get('get_default_group', lambda: None)()


def fake_new_group(ranks=None, timeout=None, backend=None, pg_options=None, **kwargs):
    if _HOOK_ENABLED:
        print(f"[HOOK] fake_new_group: ranks={ranks}", flush=True)
        group = FakeProcessGroup(
            ranks=ranks or [0], 
            rank=0 if (ranks is None or 0 in ranks) else ranks[0],
            size=len(ranks) if ranks else 1,
            backend=backend or "nccl"
        )
        return group
    return _original_funcs['new_group'](ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options, **kwargs)


def fake_destroy_process_group(group=None):
    global _DIST_INITIALIZED, _default_group
    if _HOOK_ENABLED:
        if group is None:
            _DIST_INITIALIZED = False
            _default_group = None
        return None
    return _original_funcs['destroy_process_group'](group)


def fake_get_group_rank(group, global_rank):
    if _HOOK_ENABLED:
        if isinstance(group, FakeProcessGroup) and group._ranks:
            if global_rank in group._ranks:
                return group._ranks.index(global_rank)
        return 0
    return _original_funcs.get('get_group_rank', lambda g, r: 0)(group, global_rank)


def fake_get_global_rank(group, rank):
    if _HOOK_ENABLED:
        if isinstance(group, FakeProcessGroup) and group._ranks:
            if rank < len(group._ranks):
                return group._ranks[rank]
        return rank
    return _original_funcs.get('get_global_rank', lambda g, r: r)(group, rank)


def fake_get_process_group_ranks(group):
    if _HOOK_ENABLED:
        if isinstance(group, FakeProcessGroup):
            return group._ranks
        return [0]
    return _original_funcs.get('get_process_group_ranks', lambda g: [0])(group)


# ============== 通信 Hooks ==============

def fake_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if _HOOK_ENABLED:
        if async_op:
            return FakeWork()
        return None
    return _original_funcs['all_reduce'](tensor, op, group, async_op)


def fake_all_gather(output_tensors, input_tensor, group=None, async_op=False):
    if _HOOK_ENABLED:
        for t in output_tensors:
            t.copy_(input_tensor)
        if async_op:
            return FakeWork()
        return None
    return _original_funcs['all_gather'](output_tensors, input_tensor, group, async_op)


def fake_all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    if _HOOK_ENABLED:
        chunk_size = input_tensor.numel()
        output_flat = output_tensor.view(-1)
        input_flat = input_tensor.view(-1)
        num_chunks = output_flat.numel() // chunk_size
        for i in range(num_chunks):
            output_flat[i * chunk_size:(i + 1) * chunk_size].copy_(input_flat)
        if async_op:
            return FakeWork()
        return None
    return _original_funcs['all_gather_into_tensor'](output_tensor, input_tensor, group, async_op)


def fake_reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if _HOOK_ENABLED:
        output.copy_(input_list[0])
        if async_op:
            return FakeWork()
        return None
    return _original_funcs['reduce_scatter'](output, input_list, op, group, async_op)


def fake_reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if _HOOK_ENABLED:
        chunk_size = output.numel()
        output.copy_(input.view(-1)[:chunk_size].view(output.shape))
        if async_op:
            return FakeWork()
        return None
    return _original_funcs['reduce_scatter_tensor'](output, input, op, group, async_op)


def fake_send(tensor, dst, group=None, tag=0):
    if _HOOK_ENABLED:
        return None
    return _original_funcs['send'](tensor, dst, group, tag)


def fake_recv(tensor, src=None, group=None, tag=0):
    if _HOOK_ENABLED:
        tensor.zero_()
        return None
    return _original_funcs['recv'](tensor, src, group, tag)


def fake_isend(tensor, dst, group=None, tag=0):
    if _HOOK_ENABLED:
        return FakeWork()
    return _original_funcs['isend'](tensor, dst, group, tag)


def fake_irecv(tensor, src=None, group=None, tag=0):
    if _HOOK_ENABLED:
        tensor.zero_()
        return FakeWork()
    return _original_funcs['irecv'](tensor, src, group, tag)


def fake_broadcast(tensor, src, group=None, async_op=False):
    if _HOOK_ENABLED:
        if async_op:
            return FakeWork()
        return None
    return _original_funcs['broadcast'](tensor, src, group, async_op)


def fake_broadcast_object_list(object_list, src=0, group=None, device=None):
    if _HOOK_ENABLED:
        return None
    return _original_funcs['broadcast_object_list'](object_list, src, group, device)


def fake_barrier(group=None, async_op=False, device_ids=None):
    if _HOOK_ENABLED:
        if async_op:
            return FakeWork()
        return None
    return _original_funcs['barrier'](group, async_op, device_ids)


def fake_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    if _HOOK_ENABLED:
        for i, t in enumerate(output_tensor_list):
            if i < len(input_tensor_list):
                t.copy_(input_tensor_list[i])
        if async_op:
            return FakeWork()
        return None
    return _original_funcs['all_to_all'](output_tensor_list, input_tensor_list, group, async_op)


def fake_all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, 
                           group=None, async_op=False):
    if _HOOK_ENABLED:
        output.copy_(input)
        if async_op:
            return FakeWork()
        return None
    return _original_funcs['all_to_all_single'](
        output, input, output_split_sizes, input_split_sizes, group, async_op
    )


def fake_monitored_barrier(group=None, timeout=None, wait_all_ranks=False):
    if _HOOK_ENABLED:
        return None
    return _original_funcs.get('monitored_barrier', lambda *a, **k: None)(group, timeout, wait_all_ranks)


# ============== PyNCCL Hooks ==============

# 在 communication_hook.py 中，修改 install_pynccl_hooks 函数

def install_pynccl_hooks():
    """Hook pynccl 库"""
    global _PYNCCL_HOOKED
    
    if not _HOOK_ENABLED:
        return
    
    if _PYNCCL_HOOKED:
        return
    
    print("[HOOK] Installing pynccl hooks...", flush=True)
    
    # Hook sglang.srt.distributed.device_communicators.pynccl (正确的路径)
    try:
        from sglang.srt.distributed.device_communicators import pynccl as srt_pynccl
        _original_funcs['srt_PyNcclCommunicator'] = srt_pynccl.PyNcclCommunicator
        srt_pynccl.PyNcclCommunicator = FakePyNcclCommunicator
        print("[HOOK] Hooked sglang.srt.distributed.device_communicators.pynccl.PyNcclCommunicator", flush=True)
    except ImportError as e:
        print(f"[HOOK] sglang.srt.distributed.device_communicators.pynccl not found: {e}", flush=True)
    except Exception as e:
        print(f"[HOOK] Failed to hook srt pynccl: {e}", flush=True)
    
    # 也 hook pynccl_wrapper 中的 NCCLLibrary（以防万一）
    try:
        from sglang.srt.distributed.device_communicators import pynccl_wrapper
        
        # 创建一个假的 NCCLLibrary
        class FakeNCCLLibrary:
            def __init__(self, *args, **kwargs):
                print("[HOOK] FakeNCCLLibrary.__init__", flush=True)
            
            def ncclGetVersion(self):
                return "fake-2.0.0"
            
            def ncclGetUniqueId(self):
                # 返回一个假的 unique id
                return pynccl_wrapper.ncclUniqueId()
            
            def ncclCommInitRank(self, world_size, unique_id, rank):
                print(f"[HOOK] FakeNCCLLibrary.ncclCommInitRank: world_size={world_size}, rank={rank}", flush=True)
                return None  # 返回假的 comm
            
            def ncclAllReduce(self, *args, **kwargs):
                return
            
            def ncclAllGather(self, *args, **kwargs):
                return
            
            def ncclReduceScatter(self, *args, **kwargs):
                return
            
            def ncclBroadcast(self, *args, **kwargs):
                return
            
            def ncclSend(self, *args, **kwargs):
                return
            
            def ncclRecv(self, *args, **kwargs):
                return
        
        _original_funcs['NCCLLibrary'] = pynccl_wrapper.NCCLLibrary
        pynccl_wrapper.NCCLLibrary = FakeNCCLLibrary
        print("[HOOK] Hooked pynccl_wrapper.NCCLLibrary", flush=True)
        
    except ImportError as e:
        print(f"[HOOK] pynccl_wrapper not found: {e}", flush=True)
    except Exception as e:
        print(f"[HOOK] Failed to hook pynccl_wrapper: {e}", flush=True)
    
    _PYNCCL_HOOKED = True
    print("[HOOK] pynccl hooks installed", flush=True)

# ============== 安装/卸载 ==============

def install_communication_hooks():
    """安装所有通信 hooks"""
    global _HOOK_INITIALIZED
    
    if not _HOOK_ENABLED:
        return
    
    if _HOOK_INITIALIZED:
        print("[HOOK] Communication hooks already installed", flush=True)
        return
    
    print("[HOOK] Installing communication hooks...", flush=True)
    
    # === 保存原始函数 ===
    
    # 分布式初始化
    _original_funcs['init_process_group'] = dist.init_process_group
    _original_funcs['is_initialized'] = dist.is_initialized
    _original_funcs['get_rank'] = dist.get_rank
    _original_funcs['get_world_size'] = dist.get_world_size
    _original_funcs['get_backend'] = dist.get_backend
    _original_funcs['new_group'] = dist.new_group
    _original_funcs['destroy_process_group'] = dist.destroy_process_group
    
    if hasattr(dist, 'get_default_group'):
        _original_funcs['get_default_group'] = dist.get_default_group
    if hasattr(dist, 'get_group_rank'):
        _original_funcs['get_group_rank'] = dist.get_group_rank
    if hasattr(dist, 'get_global_rank'):
        _original_funcs['get_global_rank'] = dist.get_global_rank
    if hasattr(dist, 'get_process_group_ranks'):
        _original_funcs['get_process_group_ranks'] = dist.get_process_group_ranks
    
    # 通信操作
    _original_funcs['all_reduce'] = dist.all_reduce
    _original_funcs['all_gather'] = dist.all_gather
    _original_funcs['reduce_scatter'] = dist.reduce_scatter
    _original_funcs['broadcast'] = dist.broadcast
    _original_funcs['barrier'] = dist.barrier
    _original_funcs['send'] = dist.send
    _original_funcs['recv'] = dist.recv
    _original_funcs['isend'] = dist.isend
    _original_funcs['irecv'] = dist.irecv
    
    if hasattr(dist, 'all_gather_into_tensor'):
        _original_funcs['all_gather_into_tensor'] = dist.all_gather_into_tensor
    if hasattr(dist, 'reduce_scatter_tensor'):
        _original_funcs['reduce_scatter_tensor'] = dist.reduce_scatter_tensor
    if hasattr(dist, 'all_to_all'):
        _original_funcs['all_to_all'] = dist.all_to_all
    if hasattr(dist, 'all_to_all_single'):
        _original_funcs['all_to_all_single'] = dist.all_to_all_single
    if hasattr(dist, 'broadcast_object_list'):
        _original_funcs['broadcast_object_list'] = dist.broadcast_object_list
    if hasattr(dist, 'monitored_barrier'):
        _original_funcs['monitored_barrier'] = dist.monitored_barrier
    
    # === 替换函数 ===
    
    # 分布式初始化
    dist.init_process_group = fake_init_process_group
    dist.is_initialized = fake_is_initialized
    dist.get_rank = fake_get_rank
    dist.get_world_size = fake_get_world_size
    dist.get_backend = fake_get_backend
    dist.new_group = fake_new_group
    dist.destroy_process_group = fake_destroy_process_group
    
    if hasattr(dist, 'get_default_group'):
        dist.get_default_group = fake_get_default_group
    if hasattr(dist, 'get_group_rank'):
        dist.get_group_rank = fake_get_group_rank
    if hasattr(dist, 'get_global_rank'):
        dist.get_global_rank = fake_get_global_rank
    if hasattr(dist, 'get_process_group_ranks'):
        dist.get_process_group_ranks = fake_get_process_group_ranks
    
    # 通信操作
    dist.all_reduce = fake_all_reduce
    dist.all_gather = fake_all_gather
    dist.reduce_scatter = fake_reduce_scatter
    dist.broadcast = fake_broadcast
    dist.barrier = fake_barrier
    dist.send = fake_send
    dist.recv = fake_recv
    dist.isend = fake_isend
    dist.irecv = fake_irecv
    
    if hasattr(dist, 'all_gather_into_tensor'):
        dist.all_gather_into_tensor = fake_all_gather_into_tensor
    if hasattr(dist, 'reduce_scatter_tensor'):
        dist.reduce_scatter_tensor = fake_reduce_scatter_tensor
    if hasattr(dist, 'all_to_all'):
        dist.all_to_all = fake_all_to_all
    if hasattr(dist, 'all_to_all_single'):
        dist.all_to_all_single = fake_all_to_all_single
    if hasattr(dist, 'broadcast_object_list'):
        dist.broadcast_object_list = fake_broadcast_object_list
    if hasattr(dist, 'monitored_barrier'):
        dist.monitored_barrier = fake_monitored_barrier
    
    # === 安装 PyNCCL hooks ===
    install_pynccl_hooks()
    
    _HOOK_INITIALIZED = True
    print("[HOOK] Communication hooks installed successfully", flush=True)


def uninstall_communication_hooks():
    """恢复原始通信函数"""
    global _HOOK_INITIALIZED, _DIST_INITIALIZED, _default_group, _PYNCCL_HOOKED
    
    if not _HOOK_INITIALIZED:
        return
    
    print("[HOOK] Uninstalling communication hooks...", flush=True)
    
    # 恢复所有保存的原始函数
    for name, func in _original_funcs.items():
        if name.startswith('srt_') or name.startswith('mm_'):
            # pynccl 相关的恢复
            continue
        if hasattr(dist, name):
            setattr(dist, name, func)
    
    # 恢复 pynccl
    if 'srt_PyNcclCommunicator' in _original_funcs:
        try:
            from sglang.srt.distributed import pynccl as srt_pynccl
            srt_pynccl.PyNcclCommunicator = _original_funcs['srt_PyNcclCommunicator']
        except:
            pass
    
    if 'mm_PyNcclCommunicator' in _original_funcs:
        try:
            from sglang.multimodal_gen.runtime.distributed.device_communicators import pynccl as mm_pynccl
            mm_pynccl.PyNcclCommunicator = _original_funcs['mm_PyNcclCommunicator']
        except:
            pass
    
    _HOOK_INITIALIZED = False
    _DIST_INITIALIZED = False
    _PYNCCL_HOOKED = False
    _default_group = None
    print("[HOOK] Communication hooks uninstalled", flush=True)