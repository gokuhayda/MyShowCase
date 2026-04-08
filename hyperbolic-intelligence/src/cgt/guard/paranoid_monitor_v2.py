"""
cgt/guard/paranoid_monitor_v2.py
=====================================
ParanoidMonitorV2 — Zero Trust Execution layer for v2.

Monkey-patches torch.Tensor.__add__, __iadd__, __radd__, torch.add
to intercept linear operations on manifold-tagged tensors.

This is the LAST LINE OF DEFENSE against the Tangent Addition Fallacy.

Usage
-----
    from cgt.guard import MONITOR_V2
    MONITOR_V2.install()    # patches torch globally
    MONITOR_V2.uninstall()  # restores originals

    tag_manifold_v2(tensor)    # mark tensor as on-manifold
    untag_manifold_v2(tensor)  # remove tag (after log_map)

Zero imports from legacy cgt.*.
"""

from __future__ import annotations

import threading
import traceback
import weakref
from typing import Any

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Weakref registry
# ─────────────────────────────────────────────────────────────────────────────

class _ManifoldRegistryV2:
    """Thread-safe weakref registry for v2 on-manifold tensors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def register(self, t: torch.Tensor) -> None:
        with self._lock:
            self._refs[id(t)] = t

    def unregister(self, t: torch.Tensor) -> None:
        with self._lock:
            self._refs.pop(id(t), None)

    def is_registered(self, t: torch.Tensor) -> bool:
        with self._lock:
            return id(t) in self._refs

    def clear(self) -> None:
        with self._lock:
            self._refs.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._refs)


_REGISTRY_V2 = _ManifoldRegistryV2()


def tag_manifold_v2(t: torch.Tensor) -> torch.Tensor:
    """Mark tensor as residing on the Lorentz hyperboloid (v2)."""
    _REGISTRY_V2.register(t)
    return t


def untag_manifold_v2(t: torch.Tensor) -> torch.Tensor:
    """Remove manifold tag after log_map returns to tangent space (v2)."""
    _REGISTRY_V2.unregister(t)
    return t


def is_on_manifold_v2(t: torch.Tensor) -> bool:
    return _REGISTRY_V2.is_registered(t)


# ─────────────────────────────────────────────────────────────────────────────
# ParanoidMonitor
# ─────────────────────────────────────────────────────────────────────────────

class ParanoidMonitorV2:
    """
    Intercepts torch linear operations on manifold tensors (v2 isolated).

    Patched ops:
        torch.Tensor.__add__
        torch.Tensor.__iadd__
        torch.Tensor.__radd__
        torch.add
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self._installed = False
        self._originals: dict[str, Any] = {}

    def install(self) -> None:
        """Monkey-patch torch globally. Idempotent."""
        if self._installed:
            return

        self._originals["Tensor.__add__"]  = torch.Tensor.__add__
        self._originals["Tensor.__iadd__"] = torch.Tensor.__iadd__
        self._originals["Tensor.__radd__"] = torch.Tensor.__radd__
        self._originals["torch.add"]       = torch.add

        monitor = self

        def _guarded_add(self_t: torch.Tensor, other: Any) -> torch.Tensor:
            monitor._check("__add__", self_t)
            if isinstance(other, torch.Tensor):
                monitor._check("__add__ (rhs)", other)
            return monitor._originals["Tensor.__add__"](self_t, other)

        def _guarded_iadd(self_t: torch.Tensor, other: Any) -> torch.Tensor:
            monitor._check("__iadd__", self_t)
            if isinstance(other, torch.Tensor):
                monitor._check("__iadd__ (rhs)", other)
            return monitor._originals["Tensor.__iadd__"](self_t, other)

        def _guarded_radd(self_t: torch.Tensor, other: Any) -> torch.Tensor:
            monitor._check("__radd__", self_t)
            return monitor._originals["Tensor.__radd__"](self_t, other)

        def _guarded_torch_add(input: Any, other: Any, **kwargs: Any) -> torch.Tensor:
            if isinstance(input, torch.Tensor):
                monitor._check("torch.add (input)", input)
            if isinstance(other, torch.Tensor):
                monitor._check("torch.add (other)", other)
            return monitor._originals["torch.add"](input, other, **kwargs)

        torch.Tensor.__add__  = _guarded_add    # type: ignore[method-assign]
        torch.Tensor.__iadd__ = _guarded_iadd   # type: ignore[method-assign]
        torch.Tensor.__radd__ = _guarded_radd   # type: ignore[method-assign]
        torch.add             = _guarded_torch_add  # type: ignore[assignment]

        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        torch.Tensor.__add__  = self._originals["Tensor.__add__"]   # type: ignore
        torch.Tensor.__iadd__ = self._originals["Tensor.__iadd__"]  # type: ignore
        torch.Tensor.__radd__ = self._originals["Tensor.__radd__"]  # type: ignore
        torch.add             = self._originals["torch.add"]         # type: ignore
        self._installed = False

    def _check(self, op: str, t: torch.Tensor) -> None:
        if is_on_manifold_v2(t):
            msg = (
                f"[ParanoidMonitorV2] ZERO TRUST VIOLATION: '{op}' on manifold tensor "
                f"(shape={tuple(t.shape)}, dtype={t.dtype}). "
                f"Linear ops on the Lorentz hyperboloid are GEOMETRICALLY INVALID. "
                f"Use substrate.exp_map(x, v) instead."
            )
            if self.debug:
                print(f"\n{'='*60}\n{msg}\nTraceback:")
                traceback.print_stack()
                print("=" * 60)
            raise RuntimeError(msg)

    @property
    def installed(self) -> bool:
        return self._installed


# Default global instance
MONITOR_V2 = ParanoidMonitorV2(debug=False)
