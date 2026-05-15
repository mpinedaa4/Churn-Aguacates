"""
memory_guard.py — Lightweight RAM monitoring utilities.

Provides:
  • log_memory()          – log current RSS and available RAM
  • check_memory()        – raise MemoryError if free RAM falls below threshold
  • MemoryGuard (context) – wraps a block, logs before/after, optional check
"""

import os
import logging

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

log = logging.getLogger(__name__)

# Default threshold: warn if free RAM drops below this many GB
_DEFAULT_FREE_GB_THRESHOLD = 2.0


def _mem_info() -> dict:
    """Return a dict with rss_gb, available_gb, percent_used (or zeros)."""
    if not _HAS_PSUTIL:
        return {"rss_gb": 0.0, "available_gb": 0.0, "percent_used": 0.0}
    vm  = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    rss  = proc.memory_info().rss / 1024 ** 3
    return {
        "rss_gb":       round(rss, 2),
        "available_gb": round(vm.available / 1024 ** 3, 2),
        "percent_used": round(vm.percent, 1),
    }


def log_memory(label: str = "") -> None:
    """Log current process RSS and system free RAM."""
    info = _mem_info()
    msg  = (
        f"[MEM] {label} | "
        f"Process RSS: {info['rss_gb']:.2f} GB | "
        f"System free: {info['available_gb']:.2f} GB | "
        f"Used: {info['percent_used']}%"
    )
    log.info(msg)


def check_memory(threshold_gb: float = _DEFAULT_FREE_GB_THRESHOLD, label: str = "") -> None:
    """
    Log current memory and raise MemoryError if free RAM < threshold_gb.
    Set threshold_gb=0 to skip the check (just log).
    """
    log_memory(label)
    if not _HAS_PSUTIL or threshold_gb <= 0:
        return
    info = _mem_info()
    if info["available_gb"] < threshold_gb:
        raise MemoryError(
            f"[MEM] Free RAM ({info['available_gb']:.2f} GB) is below the "
            f"safety threshold ({threshold_gb:.1f} GB). "
            f"Aborting '{label}' to prevent OOM crash."
        )


class MemoryGuard:
    """
    Context manager that logs memory before and after a block.

    Example
    -------
        with MemoryGuard("Agglomerative Clustering", threshold_gb=4):
            agg.fit_predict(X)
    """

    def __init__(self, label: str, threshold_gb: float = _DEFAULT_FREE_GB_THRESHOLD):
        self.label = label
        self.threshold_gb = threshold_gb

    def __enter__(self):
        check_memory(self.threshold_gb, f"BEFORE {self.label}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log_memory(f"AFTER  {self.label}")
        return False   # do not suppress exceptions
