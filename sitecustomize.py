"""Startup hook — activated when ``SGLANG_FAKE_BACKEND=1`` is set.

Place this file in Python's ``site-packages/`` or add its parent directory
to ``PYTHONPATH``.  Python automatically runs ``sitecustomize.py`` at
startup (before any user code).

When the env var is present, fake-backend hooks are registered with
SGLang's ``HookRegistry``.  The actual patching happens later when
``load_plugins()`` calls ``HookRegistry.apply_hooks()``, which is invoked
early in both the main process (``Engine.__init__``) and every forked
scheduler / detokenizer subprocess (``run_scheduler_process``).
"""

import os
import sys


def _is_fake_backend_enabled() -> bool:
    return os.environ.get("SGLANG_FAKE_BACKEND", "") in ("1", "true", "yes")


def _try_apply():
    if not _is_fake_backend_enabled():
        return

    try:
        from sglang_hook.hooks import apply_all

        apply_all()
    except ImportError:
        pass
    except Exception:
        import traceback

        traceback.print_exc()


_try_apply()
