"""Restrict PDB set for Step 3 when using job arrays (`--jobid` / `--totalNrJobs`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runner import Preprocessor


def apply_job_slice(pre: Preprocessor) -> None:
    if not pre.totalNrJobs:
        return
    pdblist = pre.trajectory.pdb_ids()
    pdbids_per_job = len(pdblist) // pre.totalNrJobs + 1
    jobid = pre.jobid
    assert jobid is not None
    if jobid < pre.totalNrJobs - 1:
        lo, hi = jobid * pdbids_per_job, (jobid + 1) * pdbids_per_job
        pdbids_c = [pdblist[i] for i in range(lo, hi)]
    else:
        pdbids_c = [pdblist[i] for i in range(jobid * pdbids_per_job, len(pdblist))]
    pre.trajectory = pre.trajectory.filter_to(pdbids_c)
