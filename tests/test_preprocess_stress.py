"""Opt-in end-to-end stress: requires real input data and significant time.

Run examples::

  STRESS_SEED_INPUT=/path/to/batch pytest tests/test_preprocess_stress.py -m stress -s

  STRESS_SEED_INPUT=... STRESS_FRAMES=5,15 STRESS_COPIES=4 STRESS_CORES=1,4 pytest ... -m stress -s

Environment (all optional except STRESS_SEED_INPUT):
  STRESS_SEED_INPUT   — batch directory (default: same hardcoded default as the harness)
  STRESS_ALL          — if ``1``, all PDBs in the batch
  STRESS_FRAMES       — comma list (default ``5``)
  STRESS_COPIES       — duplicate PDB entries sharing one H5 (default ``1``)
  STRESS_CORES        — comma list, default is harness sweep
  STRESS_VERIFY       — if ``1``, compare deltaforces: cores=1 vs higher
  STRESS_PRIOR        — default ``CA_DNA`` (harness default); use ``CA`` for protein-only
  STRESS_WITH_DNA_PDB — if set, a second run with this one pdbid (DNA in H5) after the first pass
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Module lives under base_model/ (package ``preprocess`` on PYTHONPATH).
_BASE = Path(__file__).resolve().parent.parent


@pytest.mark.stress
def test_preprocess_stress_harness() -> None:
    from preprocess.stress_harness import (
        DEFAULT_STRESS_INPUT_DIR,
        DEFAULT_STRESS_PRIOR,
        stress_matrix,
    )

    seed = os.environ.get("STRESS_SEED_INPUT", DEFAULT_STRESS_INPUT_DIR)
    if not Path(seed).is_dir():
        pytest.skip("Set STRESS_SEED_INPUT to a valid batch directory (or mount the default).")

    frames = [
        int(x.strip())
        for x in os.environ.get("STRESS_FRAMES", "5").split(",")
        if x.strip()
    ]
    copies = int(os.environ.get("STRESS_COPIES", "1"))
    extra = list(
        x
        for x in os.environ.get("STRESS_EXTRA", "").split()
        if x.strip()
    )
    ver = os.environ.get("STRESS_VERIFY", "") in ("1", "true", "yes")
    cr = os.environ.get("STRESS_CORES", "").strip()
    cores = [int(x) for x in cr.split(",") if x.strip()] if cr else None
    rpt = int(os.environ.get("STRESS_REPEAT", "1"))
    pids = os.environ.get("STRESS_PDBIDS", "").strip()
    pfilter = [x.strip() for x in pids.split(",") if x.strip()] if pids else None
    all_pdbs = os.environ.get("STRESS_ALL", "").lower() in ("1", "true", "yes")

    pr = os.environ.get("STRESS_PRIOR", DEFAULT_STRESS_PRIOR)
    dna_pdb = os.environ.get("STRESS_WITH_DNA_PDB", "").strip()

    stress_matrix(
        _BASE,
        Path(seed).resolve(),
        prior=pr,
        copies=copies,
        frames_list=frames,
        cores_list=cores,
        repeat=max(1, rpt),
        verify_parallel=ver,
        extra=extra,
        pdbid_filter=pfilter,
        entire_input_dir=all_pdbs,
    )
    if dna_pdb and not all_pdbs:
        stress_matrix(
            _BASE,
            Path(seed).resolve(),
            prior=pr,
            copies=copies,
            frames_list=frames,
            cores_list=cores,
            repeat=max(1, rpt),
            verify_parallel=ver,
            extra=extra,
            pdbid_filter=[dna_pdb],
            entire_input_dir=False,
        )
