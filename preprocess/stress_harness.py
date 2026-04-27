"""End-to-end preprocess stress driver (parallel pools + inner classical workers + I/O).

From ``base_model/`` a single default run uses the **protein–DNA** prior (``--prior CA_DNA``)
and **1b3t** from the 0119 batch; outputs are under the system temp only:

  python -m preprocess.stress_harness

To stress **DNA** mapping (a structure that actually has DA/DT/DG/DC in the H5), run a
second pass with a pdb from your batch, e.g.:

  python -m preprocess.stress_harness --with-dna-pdb 1xyz

(Replace ``1xyz`` with a real id in ``--seed-input`` that contains DNA in the trajectory.)

Protein-only prior (no DNA terms path): use ``--prior CA``.

Override batch dir or pdbs: ``--pdbids 1b3t,1d02`` or ``--all-pdbs``.

Environment: set ``STRESS_MAX_CORES`` to cap the default core sweep. Use ``--repeat 3`` to
detect flaky timing. If ``Too many open files`` appears, raise ``ulimit -n`` before running
large ``--copies`` / high ``--cores`` sweeps.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .loaders import gen_input_mapping

# Default one-command inputs (1b3t under 0119_successes, CA_DNA to match 0225/0120-style runs).
DEFAULT_STRESS_INPUT_DIR = "/media/DATA_18_TB_1/akshitha/0119_successes"
DEFAULT_STRESS_PDBIDS = ("1b3t",)
DEFAULT_STRESS_PRIOR = "CA_DNA"


def _first_valid_pdb_h5(input_root: Path) -> tuple[str, Path]:
    m = gen_input_mapping([{"path": os.fspath(input_root)}])
    if not m:
        raise FileNotFoundError(
            f"No result/output_*.h5 under {input_root} (see gen_input_mapping)."
        )
    sid = next(iter(m.keys()))
    return sid, Path(m[sid])


def _make_multi_pdb_input(
    input_root: Path, seed_id: str, h5: Path, n_copies: int, work: Path
) -> Path:
    if n_copies < 1:
        raise ValueError("n_copies must be >= 1")
    if n_copies == 1:
        return input_root
    h5 = h5.resolve()
    for i in range(n_copies):
        name = f"stress_{i:04d}"
        d = work / name / "result"
        d.mkdir(parents=True, exist_ok=True)
        link = d / f"output_{name}.h5"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(h5)
    return work


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _validate_run(out: Path, pdbids: list[str]) -> None:
    okp = out / "result" / "ok_list.txt"
    if not okp.is_file():
        raise AssertionError(f"Missing {okp}")
    lines = [x.strip() for x in okp.read_text(encoding="utf-8").splitlines() if x.strip()]
    if set(lines) != set(pdbids):
        raise AssertionError(f"ok_list mismatch: {lines} vs {pdbids}")
    for pid in pdbids:
        dfn = out / pid / "raw" / "deltaforces.npy"
        if not dfn.is_file():
            raise AssertionError(f"Missing {dfn}")


def _load_delta(path: Path) -> np.ndarray:
    return np.load(os.fspath(path))


def _assert_deltas_close(a: Path, b: Path, *, rtol: float, atol: float) -> None:
    xa, xb = _load_delta(a), _load_delta(b)
    if xa.shape != xb.shape:
        raise AssertionError(f"Shape mismatch {a} {xa.shape} vs {b} {xb.shape}")
    if not np.allclose(xa, xb, rtol=rtol, atol=atol):
        d = float(np.max(np.abs(xa - xb)))
        raise AssertionError(f"deltaforces differ: {a} vs {b} max_abs={d}")


def _run_subprocess(
    base_model: Path, cmd: list[str]
) -> tuple[int, float]:
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        cwd=os.fspath(base_model),
        env={**os.environ, "PYTHONPATH": os.fspath(base_model)},
        check=False,
    )
    return p.returncode, time.perf_counter() - t0


def _cap_cores(cores: Sequence[int], cap: int | None) -> list[int]:
    if cap is None:
        if os.environ.get("STRESS_MAX_CORES"):
            cap = int(os.environ["STRESS_MAX_CORES"])
        else:
            cap = (os.cpu_count() or 4) * 2
    return [c for c in sorted(set(cores)) if 1 <= c <= cap]


def run_one(
    base_model: Path,
    input_dir: Path,
    out_dir: Path,
    *,
    prior: str,
    num_frames: int,
    num_cores: int,
    extra: list[str],
    subprocess_pdbids: list[str] | None = None,
) -> float:
    cmd = [
        sys.executable,
        "-m",
        "preprocess",
        os.fspath(input_dir),
        "-o",
        os.fspath(out_dir),
        "--prior",
        prior,
        "--num-frames",
        str(num_frames),
        "--num-cores",
        str(num_cores),
        "--no-prior-plots",
    ]
    if subprocess_pdbids:
        cmd += ["--pdbids", *subprocess_pdbids]
    cmd += extra
    print("+", " ".join(cmd), flush=True)
    code, wall = _run_subprocess(base_model, cmd)
    if code != 0:
        raise RuntimeError(f"preprocess failed with {code} after {wall:.2f}s")
    return wall


def stress_matrix(
    base_model: Path,
    seed_input: Path,
    *,
    prior: str = DEFAULT_STRESS_PRIOR,
    copies: int = 1,
    frames_list: list[int] | None = None,
    cores_list: list[int] | None = None,
    repeat: int = 1,
    verify_parallel: bool = False,
    extra: list[str] | None = None,
    keep_work: bool = False,
    pdbid_filter: list[str] | None = None,
    entire_input_dir: bool = False,
) -> None:
    """Run preprocess over (frames × cores × repeat) and check outputs; optional 1 vs N-core agreement."""
    if entire_input_dir:
        f_pdb: list[str] | None = None
    elif pdbid_filter is not None:
        f_pdb = list(pdbid_filter)
    else:
        f_pdb = list(DEFAULT_STRESS_PDBIDS)
    if verify_parallel and repeat > 1:
        print("verify_parallel: forcing repeat=1 (comparing single runs per (frames, cores)).", flush=True)
        repeat = 1
    frames_list = frames_list or [5, 20]
    max_cpu = os.cpu_count() or 4
    raw_cores = cores_list or _cap_cores([1, 2, 4, max(8, max_cpu // 2), max_cpu], None)
    if verify_parallel:
        if 1 not in raw_cores:
            raise ValueError("verify_parallel requires --cores to include 1 (serial PDB pool baseline).")
        cores_list = [1] + [c for c in sorted(set(raw_cores)) if c != 1]
    else:
        cores_list = list(raw_cores)
    extra = extra or []

    work_root = (
        Path(tempfile.mkdtemp(prefix="preprocess_stress_in_"))
        if copies > 1
        else seed_input
    )

    try:
        full_map = gen_input_mapping([{"path": os.fspath(seed_input)}])
        if not full_map:
            raise FileNotFoundError(
                f"No result/output_*.h5 under {seed_input} (see gen_input_mapping)."
            )
        if f_pdb is not None:
            missing = set(f_pdb) - set(full_map.keys())
            if missing:
                raise ValueError(f"pdbids not found under input: {sorted(missing)}")
            seed_id = f_pdb[0]
            h5 = Path(full_map[seed_id])
        else:
            if len(full_map) > 1:
                print(
                    f"WARNING: processing all {len(full_map)} pdbs under {seed_input}.",
                    flush=True,
                )
            seed_id, h5 = _first_valid_pdb_h5(seed_input)
        print(f"Using seed {seed_id!r} -> {h5}", flush=True)
        input_dir = (
            _make_multi_pdb_input(seed_input, seed_id, h5, copies, work_root)
            if copies > 1
            else seed_input
        )
        discovered = list(
            gen_input_mapping([{"path": os.fspath(input_dir)}]).keys()
        )
        if copies > 1:
            subprocess_pdbids: list[str] | None = None
            pdbids = discovered
        elif f_pdb is not None:
            subprocess_pdbids = list(f_pdb)
            pdbids = list(f_pdb)
        else:
            subprocess_pdbids = None
            pdbids = discovered
        print(
            f"PDB ids in run ({len(pdbids)}): {pdbids[:5]}{'...' if len(pdbids) > 5 else ''}",
            flush=True,
        )

        for n_fr in frames_list:
            if not verify_parallel:
                for n_co in cores_list:
                    times: list[float] = []
                    for r in range(repeat):
                        out = Path(
                            tempfile.mkdtemp(
                                prefix=f"out_{n_fr}f_{n_co}c_r{r}_",
                            )
                        )
                        try:
                            t = run_one(
                                base_model,
                                input_dir,
                                out,
                                prior=prior,
                                num_frames=n_fr,
                                num_cores=n_co,
                                extra=extra,
                                subprocess_pdbids=subprocess_pdbids,
                            )
                            _validate_run(out, pdbids)
                            times.append(t)
                        finally:
                            shutil.rmtree(out, ignore_errors=True)
                    mean = statistics.mean(times)
                    s = f"frames={n_fr} cores={n_co}: " + (
                        f"wall_s={times[0]:.2f}"
                        if repeat == 1
                        else f"wall_s mean={mean:.2f} stdev={statistics.pstdev(times):.3f} runs={times}"
                    )
                    print(s, flush=True)
            else:
                out_ref = Path(
                    tempfile.mkdtemp(prefix=f"ref_{n_fr}f_1c_"),
                )
                try:
                    t_ref = run_one(
                        base_model,
                        input_dir,
                        out_ref,
                        prior=prior,
                        num_frames=n_fr,
                        num_cores=1,
                        extra=extra,
                        subprocess_pdbids=subprocess_pdbids,
                    )
                    _validate_run(out_ref, pdbids)
                    print(
                        f"frames={n_fr} cores=1: wall_s={t_ref:.2f} (reference for deltaforces check)",
                        flush=True,
                    )
                except Exception:
                    shutil.rmtree(out_ref, ignore_errors=True)
                    raise
                for n_co in [c for c in cores_list if c != 1]:
                    out_par = Path(
                        tempfile.mkdtemp(prefix=f"out_{n_fr}f_{n_co}c_"),
                    )
                    try:
                        t = run_one(
                            base_model,
                            input_dir,
                            out_par,
                            prior=prior,
                            num_frames=n_fr,
                            num_cores=n_co,
                            extra=extra,
                            subprocess_pdbids=subprocess_pdbids,
                        )
                        _validate_run(out_par, pdbids)
                        for pid in pdbids:
                            _assert_deltas_close(
                                out_ref / pid / "raw" / "deltaforces.npy",
                                out_par / pid / "raw" / "deltaforces.npy",
                                rtol=1e-5,
                                atol=1e-6,
                            )
                        print(
                            f"frames={n_fr} cores={n_co}: wall_s={t:.2f} (deltaforces match ref)",
                            flush=True,
                        )
                    finally:
                        shutil.rmtree(out_par, ignore_errors=True)
                shutil.rmtree(out_ref, ignore_errors=True)
    finally:
        if copies > 1 and not keep_work and work_root != seed_input and work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Stress-test full preprocess (optional multi-PDB from one H5).",
    )
    ap.add_argument(
        "seed_input",
        type=Path,
        nargs="?",
        default=Path(DEFAULT_STRESS_INPUT_DIR),
        help=(
            f"Input batch directory (default: {DEFAULT_STRESS_INPUT_DIR} "
            f"— then only {','.join(DEFAULT_STRESS_PDBIDS)} and prior {DEFAULT_STRESS_PRIOR} "
            f"unless --all-pdbs or --pdbids)"
        ),
    )
    ap.add_argument(
        "--all-pdbs",
        action="store_true",
        help="Run on every system under the batch dir (overrides default 1b3t).",
    )
    ap.add_argument(
        "--pdbids",
        type=str,
        default="",
        help="Comma-separated pdbids, e.g. 1b3t,1d02. Empty uses the default 1b3t (ignored with --all-pdbs).",
    )
    ap.add_argument(
        "--base-model",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to base_model (default: parent of package preprocess)",
    )
    ap.add_argument(
        "--prior",
        type=str,
        default=DEFAULT_STRESS_PRIOR,
        help=f"Preprocess --prior (default: {DEFAULT_STRESS_PRIOR} for mixed protein–DNA; use CA for protein only).",
    )
    ap.add_argument(
        "--with-dna-pdb",
        type=str,
        default="",
        metavar="PDBID",
        help="After the default pass, run the same stress matrix for this single pdbid (use a batch entry whose H5 has DNA) to stress DA/DT/DG/DC mapping.",
    )
    ap.add_argument(
        "--copies",
        type=int,
        default=1,
        help="If >1, build unique pdb dirs in a temp area, all sharing the same H5 (stress step 1/3 pools).",
    )
    ap.add_argument(
        "--frames",
        type=str,
        default="5,20",
        help="Comma-separated frame counts (e.g. 5,20,100)",
    )
    ap.add_argument(
        "--cores",
        type=str,
        default="",
        help="Comma-separated core counts; default sweep 1,2,4,… up to STRESS_MAX_CORES or cpu_count",
    )
    ap.add_argument("--repeat", type=int, default=1, help="Repeat each (frames,cores) for flakiness")
    ap.add_argument(
        "--verify-parallel",
        action="store_true",
        help="After a num-cores=1 run, require deltaforces to match under higher core counts (same data).",
    )
    ap.add_argument(
        "--keep-work",
        action="store_true",
        help="If --copies>1, do not delete the mirrored input temp dir (debug).",
    )
    ap.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args to preprocess, after --, e.g. -- --no-box",
    )
    args = ap.parse_args()
    extra: list[str] = []
    if args.extra and args.extra[0] == "--":
        extra = list(args.extra[1:])

    fr = _parse_int_list(args.frames)
    cores: list[int] | None
    if args.cores.strip():
        cores = _parse_int_list(args.cores)
    else:
        cores = None

    with_dna = args.with_dna_pdb.strip()

    if args.all_pdbs:
        pfilter: list[str] | None = None
        entire = True
    elif args.pdbids.strip():
        pfilter = [x.strip() for x in args.pdbids.split(",") if x.strip()]
        entire = False
    else:
        pfilter = None
        entire = False

    def _one_matrix(pf: list[str] | None, ent: bool) -> None:
        stress_matrix(
            args.base_model.resolve(),
            args.seed_input.resolve(),
            prior=args.prior,
            copies=args.copies,
            frames_list=fr,
            cores_list=cores,
            repeat=max(1, args.repeat),
            verify_parallel=args.verify_parallel,
            extra=extra,
            keep_work=args.keep_work,
            pdbid_filter=pf,
            entire_input_dir=ent,
        )

    _one_matrix(pfilter, entire)
    if with_dna and not ent:
        print(
            f"\n[stress_harness] Second pass: --prior {args.prior} --pdbids {with_dna} (DNA trajectory stress)\n",
            flush=True,
        )
        _one_matrix([with_dna], False)


if __name__ == "__main__":
    _main()
