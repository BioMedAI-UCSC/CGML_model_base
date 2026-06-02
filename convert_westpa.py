"""Convert one WESTPA propagation run into an h5 trajectory file with
per-segment walker weights, in the format consumed by preprocess.py.

Output layout (single h5 per pdbid):
    coordinates : (N_frames, N_atoms, 3) float32  in angstroms
    forces      : (N_frames, N_atoms, 3) float32  in kJ/mol/nm
    time        : (N_frames,) float32             frame index (monotonic)
    weight      : (N_frames,) float32             walker weight per frame
    cell_lengths/cell_angles  : carried through from the segment trajectories
                                if present.

The h5 layout matches the mdtraj HDF5TrajectoryFile format that
preprocess.load_h5_traj_slice expects. With the `weight` dataset present,
preprocess.process_step1 will write a per-bead forces_weights.npy alongside
forces.npy, and train.py --use-force-weights will multiply the FM loss
element-wise by those weights.

Parallelism: segment loading (DCD + NPZ) is I/O bound; we use a process pool
to load segments concurrently and stream them into the single h5 writer.
Resume support: if the output file already exists with the same set of
segments, the run is skipped.

Designed to handle a SINGLE WESTPA run. For multiple WESTPA runs per pdbid
(e.g. different starting points), produce one h5 per run with this script and
combine them with merge_westpa_h5.py.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import h5py
import mdtraj as md
import numpy as np
import tqdm


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert one WESTPA propagation run to a weighted h5 dataset."
    )
    parser.add_argument(
        "--westpa-dir", required=True,
        help="Path to the WESTPA propagation directory containing west.h5 and traj_segs/.",
    )
    parser.add_argument(
        "--base-traj", required=True,
        help="Base topology file (PDB) used as the mdtraj topology. Selection 'protein' is applied.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .h5 path.",
    )
    parser.add_argument(
        "--protein-name", required=True,
        help="Identifier used in log messages and stored in the output as an attribute.",
    )
    parser.add_argument(
        "--skip-iters", type=int, default=1,
        help="WESTPA iterations <= this are skipped (default 1; the initial iteration is usually a smoke iter).",
    )
    parser.add_argument(
        "--stop-iter", type=int, default=None,
        help="Last iteration to include (inclusive). Default: all iterations in west.h5.",
    )
    parser.add_argument(
        "--min-weight", type=float, default=1e-5,
        help="Drop walkers with weight below this threshold (default 1e-5). Prevents noise-floor walkers from dominating numerics.",
    )
    parser.add_argument(
        "--atom-selection", default="protein",
        help="mdtraj atom selection applied to base-traj to obtain the kept atom set (default 'protein').",
    )
    parser.add_argument(
        "--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of parallel segment loaders.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-convert even if the output exists.",
    )
    return parser.parse_args()


def _segment_paths(westpa_dir: str, iter_num: int, walker_idx: int) -> tuple[str, str]:
    seg_dir = os.path.join(westpa_dir, "traj_segs", f"{iter_num:06}", f"{walker_idx:06}")
    return os.path.join(seg_dir, "seg.dcd"), os.path.join(seg_dir, "seg.npz")


_GLOBAL_TOPOLOGY = None  # set in worker via _init_worker
_GLOBAL_SEL = None       # atom indices into the full topology; None = keep all

def _init_worker(topology_pickle_path: str, sel_indices):
    """Each worker loads the (full, pdbfixed) topology once and stores the
    output-slice indices. mdtraj topology is not picklable; we round-trip via
    pdb so workers don't reload per-segment."""
    global _GLOBAL_TOPOLOGY, _GLOBAL_SEL
    _GLOBAL_TOPOLOGY = md.load_topology(topology_pickle_path)
    _GLOBAL_SEL = None if sel_indices is None else np.asarray(sel_indices, dtype=np.int64)


def _load_segment(task: tuple):
    """Worker: load one (dcd, npz) segment, slice to output atoms, return
    (idx, coords_A, forces, n_frames, cell_lengths, cell_angles)."""
    seq_idx, iter_num, walker_idx, weight, dcd_path, npz_path = task
    seg_traj = md.load(dcd_path, top=_GLOBAL_TOPOLOGY)
    coords_full = (seg_traj.xyz * 10.0).astype(np.float32)       # nm -> angstrom
    seg_data = np.load(npz_path)
    forces_full = seg_data["forces"].astype(np.float32)
    if forces_full.shape[0] != coords_full.shape[0]:
        raise RuntimeError(
            f"iter={iter_num} walker={walker_idx}: frame count mismatch "
            f"coords={coords_full.shape[0]} forces={forces_full.shape[0]}"
        )
    if forces_full.shape[1] != coords_full.shape[1]:
        raise RuntimeError(
            f"iter={iter_num} walker={walker_idx}: atom count mismatch "
            f"coords={coords_full.shape[1]} forces={forces_full.shape[1]}"
        )
    if _GLOBAL_SEL is not None:
        coords = coords_full[:, _GLOBAL_SEL, :]
        forces = forces_full[:, _GLOBAL_SEL, :]
    else:
        coords, forces = coords_full, forces_full
    n_frames = coords.shape[0]
    cell_lengths = seg_traj.unitcell_lengths.astype(np.float32) if seg_traj.unitcell_lengths is not None else None
    cell_angles = seg_traj.unitcell_angles.astype(np.float32) if seg_traj.unitcell_angles is not None else None
    return seq_idx, weight, n_frames, coords, forces, cell_lengths, cell_angles


def _build_expanded_topology(base_pdb_path: str, ph: float = 7.0):
    """Mirror the propagator's PDBFixer step (findMissingResidues/Atoms +
    addMissingAtoms + addMissingHydrogens) so the resulting mdtraj topology
    has the same atom count as the trajectory DCDs written by OpenMM."""
    try:
        from pdbfixer import PDBFixer
    except ImportError as e:
        raise ImportError(
            "pdbfixer is required to expand the base PDB to the simulated topology."
        ) from e
    import tempfile
    from openmm.app import PDBFile as _PDBFile
    fixer = PDBFixer(filename=base_pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
        tmp_path = tf.name
    with open(tmp_path, "w") as fh:
        _PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    expanded = md.load(tmp_path)
    os.unlink(tmp_path)
    return expanded


def _build_segment_tasks(west_h5_path: str, westpa_dir: str, skip_iters: int,
                          stop_iter: Optional[int], min_weight: float):
    """Use the westpa.analysis Run reader to enumerate (iter, walker, weight).
    Returns a list of (seq_idx, iter_num, walker_idx, weight, dcd_path, npz_path)
    sorted in chronological (iter, walker) order so the output is deterministic.
    """
    try:
        from westpa.analysis import Run
    except ImportError as e:
        raise ImportError(
            "westpa is required for convert_westpa.py. Install with "
            "`pip install westpa` in your env."
        ) from e

    run = Run.open(west_h5_path)
    if stop_iter is None:
        stop_iter = len(run.iterations)

    tasks = []
    skipped = 0
    for iteration in run.iterations:
        if iteration.number <= skip_iters or iteration.number > stop_iter:
            continue
        for walker in iteration.walkers:
            if walker.weight < min_weight:
                skipped += 1
                continue
            dcd_path, npz_path = _segment_paths(westpa_dir, iteration.number, walker.index)
            tasks.append((
                len(tasks),
                iteration.number, walker.index,
                float(walker.weight), dcd_path, npz_path,
            ))
    run.close() if hasattr(run, "close") else None
    return tasks, skipped, stop_iter


def main():
    args = _parse_args()

    if os.path.exists(args.output) and not args.force:
        # Quick sanity check: do we already have a coords+forces+weight h5?
        try:
            with h5py.File(args.output, "r") as f:
                if all(k in f for k in ("coordinates", "forces", "weight")):
                    print(f"[skip] {args.output} already contains coordinates+forces+weight. Use --force to overwrite.")
                    return
        except OSError:
            pass

    west_h5 = os.path.join(args.westpa_dir, "west.h5")
    if not os.path.exists(west_h5):
        sys.exit(f"west.h5 not found in {args.westpa_dir}")

    print(f"[{args.protein_name}] Building expanded topology (pdbfixer) from: {args.base_traj}")
    expanded = _build_expanded_topology(args.base_traj)
    sel = expanded.topology.select(args.atom_selection)
    if len(sel) == 0:
        sys.exit(f"atom-selection '{args.atom_selection}' matched 0 atoms in expanded {args.base_traj}")
    n_atoms = int(len(sel))
    print(f"[{args.protein_name}] expanded topology = {expanded.n_atoms} atoms; "
          f"output slice ('{args.atom_selection}') = {n_atoms} atoms")

    # Stash the FULL (pdbfixed) topology so workers can load DCDs; slicing
    # to the output atom set happens after load.
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    topo_path = args.output + ".topology.pdb"
    expanded.save_pdb(topo_path)
    sel_indices = None if n_atoms == expanded.n_atoms else sel.tolist()

    print(f"[{args.protein_name}] Enumerating segments...")
    tasks, skipped, stop_iter = _build_segment_tasks(
        west_h5, args.westpa_dir, args.skip_iters, args.stop_iter, args.min_weight,
    )
    if not tasks:
        sys.exit("No segments to convert after applying filters.")
    print(f"[{args.protein_name}] {len(tasks)} segments to convert "
          f"({skipped} dropped by --min-weight), stop_iter={stop_iter}")

    # Stage to a temp file so a crashed run never leaves a corrupt output behind.
    tmp_out = args.output + ".tmp"
    if os.path.exists(tmp_out):
        os.unlink(tmp_out)

    # Buffer results until we get the next in-order chunk so we can write
    # contiguously, but we still launch all segments in parallel.
    next_seq = 0
    pending = {}
    frame_offset = 0
    n_workers = max(1, args.num_workers)

    with h5py.File(tmp_out, "w") as fout:
        coords_dset = fout.create_dataset(
            "coordinates", shape=(0, n_atoms, 3),
            maxshape=(None, n_atoms, 3), dtype="float32", chunks=True,
        )
        forces_dset = fout.create_dataset(
            "forces", shape=(0, n_atoms, 3),
            maxshape=(None, n_atoms, 3), dtype="float32", chunks=True,
        )
        forces_dset.attrs["units"] = "kilojoules/mole/nanometer"
        weight_dset = fout.create_dataset(
            "weight", shape=(0,), maxshape=(None,), dtype="float32", chunks=True,
        )
        time_dset = fout.create_dataset(
            "time", shape=(0,), maxshape=(None,), dtype="float32", chunks=True,
        )
        cell_lengths_dset = None
        cell_angles_dset = None

        fout.attrs["protein_name"] = args.protein_name
        fout.attrs["westpa_dir"] = os.path.abspath(args.westpa_dir)
        fout.attrs["stop_iter"] = stop_iter
        fout.attrs["min_weight"] = args.min_weight
        fout.attrs["atom_selection"] = args.atom_selection

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker, initargs=(topo_path, sel_indices),
        ) as pool:
            futures = [pool.submit(_load_segment, t) for t in tasks]
            with tqdm.tqdm(total=len(tasks), desc=f"{args.protein_name}: convert") as pbar:
                for fut in as_completed(futures):
                    seq_idx, weight, n_f, coords, forces, cell_l, cell_a = fut.result()
                    pending[seq_idx] = (weight, n_f, coords, forces, cell_l, cell_a)
                    # Flush as many in-order results as we can.
                    while next_seq in pending:
                        weight, n_f, coords, forces, cell_l, cell_a = pending.pop(next_seq)
                        new_size = frame_offset + n_f
                        coords_dset.resize(new_size, axis=0)
                        forces_dset.resize(new_size, axis=0)
                        weight_dset.resize(new_size, axis=0)
                        time_dset.resize(new_size, axis=0)
                        coords_dset[frame_offset:new_size] = coords
                        forces_dset[frame_offset:new_size] = forces
                        weight_dset[frame_offset:new_size] = np.full(n_f, weight, dtype=np.float32)
                        time_dset[frame_offset:new_size] = np.arange(frame_offset, new_size, dtype=np.float32)
                        if cell_l is not None:
                            if cell_lengths_dset is None:
                                cell_lengths_dset = fout.create_dataset(
                                    "cell_lengths", shape=(0, 3),
                                    maxshape=(None, 3), dtype="float32", chunks=True,
                                )
                                cell_angles_dset = fout.create_dataset(
                                    "cell_angles", shape=(0, 3),
                                    maxshape=(None, 3), dtype="float32", chunks=True,
                                )
                            cell_lengths_dset.resize(new_size, axis=0)
                            cell_angles_dset.resize(new_size, axis=0)
                            cell_lengths_dset[frame_offset:new_size] = cell_l
                            cell_angles_dset[frame_offset:new_size] = cell_a
                        frame_offset = new_size
                        next_seq += 1
                        pbar.update(1)

    os.replace(tmp_out, args.output)

    # Companion manifest so merge_westpa_h5 / preprocess can introspect easily.
    manifest = {
        "protein_name": args.protein_name,
        "n_segments": len(tasks),
        "n_frames": frame_offset,
        "n_atoms": int(n_atoms),
        "stop_iter": stop_iter,
        "min_weight": args.min_weight,
        "westpa_dir": os.path.abspath(args.westpa_dir),
        "base_traj": os.path.abspath(args.base_traj),
        "atom_selection": args.atom_selection,
        "weight_stats": {
            "min": float(min(t[3] for t in tasks)),
            "max": float(max(t[3] for t in tasks)),
            "n_dropped_lowweight": skipped,
        },
    }
    with open(args.output + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{args.protein_name}] wrote {frame_offset} frames -> {args.output}")


if __name__ == "__main__":
    main()
