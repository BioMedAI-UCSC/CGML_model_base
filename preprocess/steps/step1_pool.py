"""Parallel Step 1 over PDBs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..worker_pool import run_pdb_pool

if TYPE_CHECKING:
    from ..runner import Preprocessor


def run_step1_parallel(pre: Preprocessor) -> None:
    if not pre.settings.do_step_1:
        return
    err = run_pdb_pool(
        pre.num_cores,
        pre.trajectory.as_dict(),
        pre.step1_threading,
        "Processing Step 1",
    )
    if err:
        print("errorList", err)
        print("errorList keys", err.keys())
