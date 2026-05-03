"""Runtime toggles for preprocessing (replaces former module-level globals)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PreprocessSettings:
    filter_not_processed_step_one: bool = False
    use_cached_fits: List[str] = field(default_factory=list)
    device_step_3: str = "cpu"
    do_step_1: bool = True
    regen_cache_files: bool = True
