"""Merge multiple per-run weighted h5 datasets into one combined dataset
ready for preprocess.py.

Use case: a single pdbid has been simulated under multiple separate WESTPA
runs (e.g. different starting conformations, different progress coordinates,
or different temperatures). Each run was converted independently with
convert_westpa.py, producing one h5 per run. This script concatenates them
into one h5 with a unified `weight` array.

Weight handling (`--normalize`):
    none      Just concatenate weights as-is. Walker probabilities from
              different runs are NOT comparable in general — use only when
              you intentionally want a single run to dominate the loss.
    per-run   (DEFAULT) Rescale each run so that mean(weight_within_run) == 1.
              All runs contribute equally on average; rare-state walkers
              keep their relative emphasis within each run.
    global    Rescale the concatenated weights so the mean is 1. Runs with
              more frames carry proportionally more weight.

The output h5 has the same schema as convert_westpa.py output, so it can
be dropped directly into preprocess.py's input directory.
"""
import argparse
import json
import os
import sys
from typing import Optional

import h5py
import numpy as np
import tqdm


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--inputs", nargs="+", required=True,
        help="One or more per-run h5 files (output of convert_westpa.py) for the SAME pdbid.",
    )
    p.add_argument(
        "--output", required=True,
        help="Combined output h5 path.",
    )
    p.add_argument(
        "--normalize", choices=("per-run", "global", "none"), default="per-run",
        help="Weight normalization across runs. Default: per-run (each run mean=1).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite the output if it exists.",
    )
    return p.parse_args()


def _load_run(path: str):
    with h5py.File(path, "r") as f:
        for key in ("coordinates", "forces", "weight"):
            if key not in f:
                raise KeyError(f"{path}: missing dataset '{key}'")
        coords = f["coordinates"][:]
        forces = f["forces"][:]
        weight = f["weight"][:]
        time = f["time"][:] if "time" in f else None
        cell_l = f["cell_lengths"][:] if "cell_lengths" in f else None
        cell_a = f["cell_angles"][:] if "cell_angles" in f else None
        force_units = f["forces"].attrs.get("units", "kilojoules/mole/nanometer")
    return coords, forces, weight, time, cell_l, cell_a, str(force_units)


def main():
    args = _parse_args()

    if os.path.exists(args.output) and not args.force:
        sys.exit(f"{args.output} already exists. Use --force to overwrite.")

    runs = []
    n_atoms_seen: Optional[int] = None
    force_units_seen: Optional[str] = None
    cell_consistency: Optional[bool] = None

    for path in args.inputs:
        print(f"[merge] loading {path}")
        coords, forces, weight, time, cell_l, cell_a, force_units = _load_run(path)
        if n_atoms_seen is None:
            n_atoms_seen = coords.shape[1]
            force_units_seen = force_units
            cell_consistency = cell_l is not None
        else:
            if coords.shape[1] != n_atoms_seen:
                sys.exit(
                    f"{path}: n_atoms={coords.shape[1]} but first run had {n_atoms_seen}. "
                    "Cannot merge runs with different atom counts."
                )
            if force_units != force_units_seen:
                sys.exit(
                    f"{path}: forces units '{force_units}' but first run had "
                    f"'{force_units_seen}'. Convert before merging."
                )
            if (cell_l is not None) != cell_consistency:
                # Allow but warn: pad missing cells with NaN-skipping logic during write.
                print(
                    f"[merge] WARN {path}: cell-info presence differs from first run; "
                    "cell info will be dropped from the merged output."
                )
                cell_consistency = False
        runs.append((path, coords, forces, weight, time, cell_l, cell_a))

    # Apply weight normalization.
    if args.normalize == "per-run":
        for i, (_path, _c, _f, w, _t, _cl, _ca) in enumerate(runs):
            m = float(w.mean()) if w.size else 0.0
            if m <= 0:
                sys.exit(f"{runs[i][0]}: mean weight is {m}, cannot normalize.")
            runs[i] = (*runs[i][:3], (w / m).astype(np.float32), *runs[i][4:])
    elif args.normalize == "global":
        all_w = np.concatenate([r[3] for r in runs])
        m = float(all_w.mean()) if all_w.size else 0.0
        if m <= 0:
            sys.exit(f"global mean weight is {m}, cannot normalize.")
        for i in range(len(runs)):
            runs[i] = (*runs[i][:3], (runs[i][3] / m).astype(np.float32), *runs[i][4:])
    # else 'none' -> leave as-is

    n_atoms = n_atoms_seen
    total_frames = sum(r[1].shape[0] for r in runs)
    print(f"[merge] {len(runs)} runs, {total_frames} total frames, {n_atoms} atoms")

    tmp_out = args.output + ".tmp"
    if os.path.exists(tmp_out):
        os.unlink(tmp_out)

    frame_offset = 0
    run_boundaries = []  # (run_path, start_frame, end_frame) for the manifest

    with h5py.File(tmp_out, "w") as fout:
        coords_dset = fout.create_dataset(
            "coordinates", shape=(total_frames, n_atoms, 3),
            dtype="float32", chunks=True,
        )
        forces_dset = fout.create_dataset(
            "forces", shape=(total_frames, n_atoms, 3),
            dtype="float32", chunks=True,
        )
        forces_dset.attrs["units"] = force_units_seen
        weight_dset = fout.create_dataset(
            "weight", shape=(total_frames,), dtype="float32", chunks=True,
        )
        time_dset = fout.create_dataset(
            "time", shape=(total_frames,), dtype="float32", chunks=True,
        )
        cell_lengths_dset = None
        cell_angles_dset = None
        if cell_consistency:
            cell_lengths_dset = fout.create_dataset(
                "cell_lengths", shape=(total_frames, 3), dtype="float32", chunks=True,
            )
            cell_angles_dset = fout.create_dataset(
                "cell_angles", shape=(total_frames, 3), dtype="float32", chunks=True,
            )

        for path, coords, forces, weight, time, cell_l, cell_a in tqdm.tqdm(runs, desc="writing"):
            n = coords.shape[0]
            sl = slice(frame_offset, frame_offset + n)
            coords_dset[sl] = coords
            forces_dset[sl] = forces
            weight_dset[sl] = weight
            if time is not None:
                time_dset[sl] = (time + frame_offset).astype(np.float32) if time[0] == 0 else time.astype(np.float32)
            else:
                time_dset[sl] = np.arange(frame_offset, frame_offset + n, dtype=np.float32)
            if cell_consistency and cell_l is not None:
                cell_lengths_dset[sl] = cell_l
                cell_angles_dset[sl] = cell_a
            run_boundaries.append((path, frame_offset, frame_offset + n))
            frame_offset += n

        fout.attrs["merged_from"] = json.dumps([
            {"path": os.path.abspath(p), "start": s, "end": e}
            for p, s, e in run_boundaries
        ])
        fout.attrs["normalize"] = args.normalize

    os.replace(tmp_out, args.output)

    manifest = {
        "output": os.path.abspath(args.output),
        "normalize": args.normalize,
        "n_runs": len(runs),
        "n_frames": total_frames,
        "n_atoms": int(n_atoms),
        "runs": [
            {"path": os.path.abspath(p), "start": s, "end": e}
            for p, s, e in run_boundaries
        ],
    }
    with open(args.output + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[merge] wrote {total_frames} frames -> {args.output}")


if __name__ == "__main__":
    main()
