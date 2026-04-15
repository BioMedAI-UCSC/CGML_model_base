from __future__ import annotations

from typing import TYPE_CHECKING

from ..worker_pool import run_pdb_pool

if TYPE_CHECKING:
    from ..runner import Preprocessor


def run_step3_parallel(pre: Preprocessor) -> None:
    pdbids = pre.trajectory.pdb_ids()
    print(f"Step 3: Processing {len(pdbids)} pdbids")
    if len(pdbids) <= 1:
        for pdbid in pdbids:
            pre.step3_threading(pdbid)
        return

    err = run_pdb_pool(
        pre.num_cores,
        pre.trajectory.as_dict(),
        pre.step3_threading,
        "Processing Step 3",
    )
    if err:
        print("step3 errorList", err)
        print("step3 errorList keys", err.keys())


def write_ok_list(pre: Preprocessor) -> None:
    with open(pre.paths.ok_list_txt(), "wt", encoding="utf-8") as ok_list:
        ok_list.write("\n".join(pre.trajectory.pdb_ids()))
