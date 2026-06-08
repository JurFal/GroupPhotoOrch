"""Stage 2 geometry and insertion-planning tools exposed to the Agent."""
from __future__ import annotations

from ..adapters.person_inserter_adapter import find_candidates as find_insertion_candidates

__all__ = ["find_insertion_candidates"]
