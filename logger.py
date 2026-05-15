"""
logger.py — Centralised logging setup for the PHQ-9 ML pipeline.

Every module imports `get_logger(__name__)` instead of calling `print()`.
Logs are written simultaneously to:
  • stdout          (coloured, for live monitoring via `tail -f`)
  • logs/pipeline.log  (plain text, permanent record)

Usage
-----
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Starting EDA...")
    log.warning("Large dataset detected — using sampled UMAP")
    log.error("Clustering failed: %s", exc)
"""

import logging
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "pipeline.log"
LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ANSI colour codes for the console handler
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColouredFormatter(logging.Formatter):
    """Formatter that prepends an ANSI colour code to the level name."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        return super().format(record)


def _setup_root_logger() -> None:
    """Configure the root logger once at import time."""
    LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    if root.handlers:          # already configured (e.g. pytest re-imports)
        return

    root.setLevel(logging.DEBUG)

    # ── File handler (plain text, always DEBUG level) ────────────────────────
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FMT, datefmt=DATE_FMT))
    root.addHandler(fh)

    # ── Console handler (coloured, INFO and above) ───────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColouredFormatter(LOG_FMT, datefmt=DATE_FMT))
    root.addHandler(ch)


_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger.  Call once per module: log = get_logger(__name__)"""
    return logging.getLogger(name)
