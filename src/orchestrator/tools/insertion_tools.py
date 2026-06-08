"""Backward-compatible insertion tool aliases.

Stage 2 geometry planning now lives in stage2_tools.py. Keep this module so old
imports and tool aliases continue to work.
"""
from __future__ import annotations

from .stage2_tools import find_insertion_candidates as find_candidates

__all__ = ["find_candidates"]
