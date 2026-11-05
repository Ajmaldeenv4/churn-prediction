"""Shared pytest setup: make `src/` importable from every test module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
