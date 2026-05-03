"""Compare two preprocess output trees (e.g. pre-refactor vs refactor) for the same pdbs.

Examples::

  python -m preprocess.compare_runs /path/to/REF /path/to/NEW --pdbids 1b3t
  # Same trajectories/forces/embeddings, ignore priorfit-dependent arrays:
  python -m preprocess.compare_runs REF NEW --pdbids 1b3t --geometry-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Prior fit + torch step-3: delta/prior_energy differ if priors.yaml differs (cgschnet vs refactor, reruns).
_GEOMETRY_FILES = ("coordinates.npy", "forces.npy", "embeddings.npy", "box.npy")
_RAW_FILES = (
    "deltaforces.npy",
    "forces.npy",
    "embeddings.npy",
    "coordinates.npy",
    "prior_energy.npy",
    "box.npy",
)


def _n_frames_in_info(d: dict) -> int | None:
    """int = first n frames; None = all (or unparseable)."""
    n = d.get("num_frames")
    if n is not None and isinstance(n, (int, float)) and n == n and not isinstance(n, bool):
        return int(n)
    t = d.get("frame_slice")
    if t is None or t == "None" or t == "":
        return None
    s = str(t)
    if s in (":::", "::", ):
        return None
    m = re.match(r"^0:([0-9]+):", s)
    if m:
        return int(m.group(1))
    m2 = re.match(r"^:([0-9]+):", s)
    if m2:
        return int(m2.group(1))
    return None


def _is_all_frames_fs(fs) -> bool:
    if fs is None or fs == "None" or fs == "":
        return True
    return str(fs).strip(":") == "" and ":" in str(fs)


def _info_frame_match(a: dict, b: dict) -> bool:
    """Cgschnet stores `num_frames` only; refactored also stores `frame_slice`."""
    na, nb = _n_frames_in_info(a), _n_frames_in_info(b)
    if na is not None or nb is not None:
        return na == nb
    fa, fb = a.get("frame_slice"), b.get("frame_slice")
    if _is_all_frames_fs(fa) and _is_all_frames_fs(fb):
        return True
    return (fa or "") == (fb or "")


def _compare_npy(
    a: Path, b: Path, *, rtol: float, atol: float
) -> tuple[str, str]:
    if not a.is_file() and not b.is_file():
        return ("skip", "both missing")
    if not a.is_file():
        return ("fail", f"ref missing: {a}")
    if not b.is_file():
        return ("fail", f"new missing: {b}")
    x, y = np.load(a), np.load(b)
    if x.shape != y.shape:
        return ("fail", f"shape {x.shape} vs {y.shape}")
    if x.dtype != y.dtype:
        if np.issubdtype(x.dtype, np.floating) and np.issubdtype(y.dtype, np.floating):
            pass
        else:
            return ("fail", f"dtype {x.dtype} vs {y.dtype}")
    d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
    if np.allclose(x, y, rtol=rtol, atol=atol):
        return ("ok", f"max|diff|={d:.6e} (within rtol/atol)")
    return ("fail", f"max_abs_diff={d:.6e} (rtol={rtol} atol={atol})")


def compare_roots(
    ref_root: Path,
    new_root: Path,
    pdbids: list[str],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    geometry_only: bool = False,
) -> int:
    any_fail = False
    files = list(_GEOMETRY_FILES) if geometry_only else list(_RAW_FILES)
    if geometry_only:
        print(
            "compare_runs: geometry-only (coordinates, forces, embeddings, box); "
            "skipping deltaforces & prior_energy (depend on fitted priors / step-3).",
            flush=True,
        )
    for pid in pdbids:
        rdir, ndir = ref_root / pid / "raw", new_root / pid / "raw"
        print(f"=== {pid} ===", flush=True)
        for name in files:
            status, msg = _compare_npy(rdir / name, ndir / name, rtol=rtol, atol=atol)
            print(f"  {name}: {status} — {msg}", flush=True)
            if status == "fail":
                any_fail = True
    ref_info, new_info = ref_root / "result" / "info.json", new_root / "result" / "info.json"
    if ref_info.is_file() and new_info.is_file():
        with open(ref_info, encoding="utf-8") as f:
            a = json.load(f)
        with open(new_info, encoding="utf-8") as f:
            b = json.load(f)
        for k in ("prior_name", "optimize_forces", "box"):
            if a.get(k) != b.get(k):
                print(f"WARNING: info.json {k!r} differs: {a.get(k)} vs {b.get(k)}", flush=True)
                any_fail = True
        if not _info_frame_match(a, b):
            print(
                f"WARNING: info.json frame selection differs: num_frames/slice "
                f"{a.get('num_frames', a.get('frame_slice'))!r} vs {b.get('num_frames', b.get('frame_slice'))!r}",
                flush=True,
            )
            any_fail = True
    if any_fail:
        print("compare_runs: FAILED", file=sys.stderr, flush=True)
        return 1
    print("compare_runs: all compared arrays match (within tolerances).", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare preprocess outputs: ref (e.g. 0225_preprocessed) vs new run.",
    )
    ap.add_argument("ref_root", type=Path, help="Reference run root (e.g. 0225_preprocessed)")
    ap.add_argument("new_root", type=Path, help="New run root to validate")
    ap.add_argument(
        "--pdbids", type=str, required=True, help="Comma-separated pdbids to compare"
    )
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument(
        "--geometry-only",
        action="store_true",
        help="Only compare coordinates, forces, embeddings, box (excludes delta/prior from prior-fit).",
    )
    args = ap.parse_args()
    ids = [x.strip() for x in args.pdbids.split(",") if x.strip()]
    raise SystemExit(
        compare_roots(
            args.ref_root.resolve(),
            args.new_root.resolve(),
            ids,
            rtol=args.rtol,
            atol=args.atol,
            geometry_only=args.geometry_only,
        )
    )


if __name__ == "__main__":
    main()
