"""Stable checkout paths; optional assets stay in the configured data directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def code_files(root=PROJECT_ROOT):
    """Actual application/experiment source files included in new freezes."""
    root = Path(root)
    return tuple(sorted(set(root.glob('*.py')) | set((root/'optical_agent').rglob('*.py'))))
