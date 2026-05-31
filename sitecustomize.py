"""Startup hook — activated when ``SGLANG_HOOK=1`` is set.

Place this file in Python's ``site-packages/`` or add its parent directory
to ``PYTHONPATH``.  Python automatically runs ``sitecustomize.py`` at
startup (before any user code).

When the env var is present, environment compatibility patches are applied
first, then fake-backend hooks are registered with SGLang's ``HookRegistry``.
"""

import os
import sys
import importlib
import importlib.util
import types


def _is_fake_backend_enabled() -> bool:
    return os.environ.get("SGLANG_HOOK", "") in ("1", "true", "yes")


def _apply_env_patches():
    """Fix environment compatibility issues before importing sglang.

    These patches address known issues with:
    - ``kernels`` package requiring ``revision``/``version``
    - ``sgl_kernel`` missing CUDA libraries
    - No accelerator (CUDA) available on CPU-only machines
    """
    try:
        import kernels.layer.layer as layer_mod
        import kernels.layer.func as func_mod

        for mod, cls_name in [
            (layer_mod, "LayerRepository"),
            (func_mod, "FuncRepository"),
        ]:
            orig = getattr(mod, cls_name).__init__

            def make_patched(o):
                def p(
                    self,
                    repo_id,
                    *,
                    layer_name=None,
                    func_name=None,
                    revision=None,
                    version=None,
                    trust_remote_code=False,
                    **kw,
                ):
                    try:
                        if layer_name is not None:
                            o(
                                self,
                                repo_id,
                                layer_name=layer_name,
                                revision=revision,
                                version=version,
                                trust_remote_code=trust_remote_code,
                                **kw,
                            )
                        else:
                            o(
                                self,
                                repo_id,
                                func_name=func_name,
                                revision=revision,
                                version=version,
                                trust_remote_code=trust_remote_code,
                                **kw,
                            )
                    except ValueError:
                        self._repo_id = repo_id
                        self._version = 1

                return p

            setattr(getattr(mod, cls_name), "__init__", make_patched(orig))
    except Exception:
        pass

    class _FakeKernelModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return lambda *a, **kw: None

    for name in [
        "sgl_kernel.kvcacheio",
        "sgl_kernel.allreduce",
        "sgl_kernel.common_ops",
        "sgl_kernel.shm_broadcast",
        "sgl_kernel.memory_pool",
    ]:
        if name not in sys.modules:
            sys.modules[name] = _FakeKernelModule(name)

    # Mock missing vllm dependency (submodule-friendly)
    class _FakeVllmModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return lambda *a, **kw: None

    for name in ["vllm", "vllm._custom_ops"]:
        if name not in sys.modules:
            sys.modules[name] = _FakeVllmModule(name)

    # If CUDA is not available, force device to cpu so that SGLang
    # can still initialise on machines with driver/PyTorch mismatches.
    try:
        import torch

        if not torch.cuda.is_available():
            import sglang.srt.utils.common as _cu

            _cu.get_device = lambda: "cpu"
            _cu.is_cuda = lambda: False
            _cu.is_cuda_alike = lambda: False
    except Exception:
        pass


def _try_apply():
    if not _is_fake_backend_enabled():
        return

    _apply_env_patches()

    try:
        from sglang_hook.hooks import apply_all

        apply_all()
    except Exception:
        import traceback

        print(
            "[sglang_hook] Failed to apply fake-backend hooks:",
            file=sys.stderr,
        )
        traceback.print_exc()


_try_apply()
