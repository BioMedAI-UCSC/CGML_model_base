from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import h5py
import mdtraj
import numpy as np
from tqdm import tqdm

from module.make_deltaforces import DeltaForces

from .loaders import load_h5_traj_slice, slice_to_str
from .paths import PreprocessPaths
from .settings import PreprocessSettings
from .trajectory_source import H5BatchTrajectorySource, TrajectorySource

_KJNM_TO_KCALA = 0.02390057361376673


class Preprocessor:
    def __init__(
        self,
        dataset_conf,
        input_path_map: Mapping[str, str],
        save_path: str | Path,
        prior_builder,
        prior_file,
        prior_name,
        frame_slice,
        temp=300,
        optimize_forces=False,
        box=True,
        prior_plots=True,
        resume_preprocess=False,
        num_cores=32,
        jobid=None,
        totalNrJobs=None,
        settings: PreprocessSettings | None = None,
        trajectory_source: TrajectorySource | None = None,
    ):
        if trajectory_source is not None:
            self.trajectory = trajectory_source
        else:
            self.trajectory = H5BatchTrajectorySource(input_path_map)

        self.dataset_conf = dataset_conf
        self.paths = PreprocessPaths(Path(save_path))
        self.save_path = os.fspath(self.paths.root)
        self.prior_builder = prior_builder
        self.prior_file = prior_file
        self.prior_name = prior_name
        self.frame_slice = frame_slice
        self.temp = temp
        self.jobid = jobid
        self.totalNrJobs = totalNrJobs
        self.settings = settings if settings is not None else PreprocessSettings()

        if self.settings.filter_not_processed_step_one:
            done = {p.parent.parent.name for p in self.paths.glob_step1_fit_ok()}
            m = {k: v for k, v in self.trajectory.as_dict().items() if k in done}
            self.trajectory = H5BatchTrajectorySource(m)
            print("%d pdbs left after removing pdbs not processed in step 1" % len(m))

        self.optimize_forces = optimize_forces
        self.box = box
        self.prior_plots = prior_plots
        self.resume_preprocess = resume_preprocess
        self.num_cores = num_cores

        print("Input directory paths:", [i["path"] for i in self.dataset_conf])
        print("Save directory path:", self.save_path)
        print(f"Temperature: {self.temp}")
        print("Frame slice:", slice_to_str(self.frame_slice))
        print("Number of cores used for parallelization:", self.num_cores)

    def _step3_classical_workers(self, n_frames: int) -> int:
        if n_frames <= 1:
            return 1
        cpu = os.cpu_count() or 4
        n_pdbs = len(self.trajectory.pdb_ids())
        if n_pdbs <= 1:
            return min(n_frames, max(1, min(self.num_cores, cpu)))
        concurrent_pdbs = min(max(1, self.num_cores), cpu)
        per_pdb = max(1, cpu // concurrent_pdbs)
        return min(n_frames, max(2, per_pdb))

    def step1_threading(self, pdbid):
        try:
            if not (self.resume_preprocess and self.paths.fit_ok(pdbid).exists()):
                bar_pos = _worker_bar_position()
                self.process_step1(pdbid, bar_pos)
                return []
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(f"{pdbid}:", e)
            raise

    def step3_threading(self, pdbid):
        try:
            bar_pos = _worker_bar_position()
            self.process_step3(pdbid, bar_pos)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(f"{pdbid}:", e)
            raise

    def preprocess(self):
        from .pipeline import run_preprocess_pipeline

        run_preprocess_pipeline(self)

    def save_data(self, pdbid, trajectory, embeddings, forces):
        raw = self.paths.pdb_raw(pdbid)
        np.save(raw / "embeddings.npy", embeddings)
        np.save(raw / "forces.npy", forces)
        np.save(raw / "coordinates.npy", trajectory.xyz)
        box_path = raw / "box.npy"
        if self.box:
            np.save(box_path, trajectory.unitcell_vectors)
        elif box_path.exists():
            box_path.unlink()

    def process_step1(self, pdbid, bar_position=0):
        with tqdm(
            total=7,
            position=bar_position,
            desc=f"{pdbid}: File path setup",
            dynamic_ncols=True,
            leave=False,
        ) as pbar:

            def tick(msg):
                pbar.update(1)
                pbar.set_description_str(f"{pdbid}: {msg}")

            self.paths.pdb_raw(pdbid).mkdir(parents=True, exist_ok=True)
            self.paths.pdb_processed(pdbid).mkdir(parents=True, exist_ok=True)

            AAtraj, path = step1_load_aa(self, pdbid, tick)
            cg_map, mol, _topology = step1_write_topology(self, pdbid, AAtraj, tick)
            forces = step1_map_forces(self, pdbid, path, AAtraj, cg_map, tick)
            cg_traj = step1_cg_traj(self, AAtraj, cg_map, tick)
            step1_save_raw(self, pdbid, cg_traj, cg_map, forces, tick)
            step1_attach_coords(mol, cg_traj)
            step1_prior_cache(self, pdbid, mol, cg_traj, tick)

    def process_step2(self):
        if self.prior_plots:
            plot_dir = self.paths.prior_fit_plots_dir()
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_dir_s = os.fspath(plot_dir)
        else:
            plot_dir_s = None

        self.prior_builder.fit(
            self.temp,
            plot_dir=plot_dir_s,
            use_cached_fits=self.settings.use_cached_fits,
        )

    def process_step3(self, pdbid, bar_position=0):
        step3_remove_stale_prior_files(self, pdbid)
        paths = step3_delta_paths(self, pdbid)
        step3_run_delta_forces(self, paths, bar_position)


# --- Step 1 helpers ---


def step1_load_aa(pre: Preprocessor, pdbid: str, tick: Callable[[str], None]) -> tuple[mdtraj.Trajectory, str]:
    tick("Loading trajectory")
    path = pre.trajectory.h5_path(pdbid)
    AAtraj = load_h5_traj_slice(path, pre.frame_slice)
    assert AAtraj.xyz is not None
    AAtraj.xyz *= 10
    return AAtraj, path


def step1_write_topology(
    pre: Preprocessor, pdbid: str, AAtraj: mdtraj.Trajectory, tick: Callable[[str], None]
) -> tuple[object, object, object]:
    tick("Building CG mapping")
    cg_map = pre.prior_builder.build_mapping(AAtraj.topology)
    mol = pre.prior_builder.make_mol(cg_map)
    topology = cg_map.to_mol(bonds=True, angles=True, dihedrals=True)
    mol.write(os.fspath(pre.paths.pdb_processed(pdbid) / f"{pdbid}_processed.psf"))
    topology.write(os.fspath(pre.paths.pdb_processed(pdbid) / "topology.psf"))
    return cg_map, mol, topology


def step1_map_forces(
    pre: Preprocessor,
    pdbid: str,
    path: str,
    AAtraj: mdtraj.Trajectory,
    cg_map: object,
    tick: Callable[[str], None],
) -> np.ndarray:
    tick("Mapping CG forces")
    with h5py.File(path, "r") as f:
        forces = f["forces"][pre.frame_slice, :, :]  # pyright: ignore[reportIndexIssue]
        if pre.optimize_forces:
            forces = cg_map.cg_optimal_forces(AAtraj, forces)
        else:
            forces = cg_map.cg_forces(forces)
        assert len(forces) == len(AAtraj)
        return forces * _KJNM_TO_KCALA


def step1_cg_traj(
    pre: Preprocessor, AAtraj: mdtraj.Trajectory, cg_map: object, tick: Callable[[str], None]
) -> mdtraj.Trajectory:
    tick("Mapping CG coordinates")
    xyz = cg_map.cg_positions(AAtraj.xyz)
    cg_traj = mdtraj.Trajectory(xyz, topology=cg_map.to_mdtraj())
    if pre.box and AAtraj.unitcell_lengths is not None:
        cg_traj.unitcell_lengths = AAtraj.unitcell_lengths * 10
        cg_traj.unitcell_angles = AAtraj.unitcell_angles
    else:
        cg_traj.unitcell_lengths = None
        cg_traj.unitcell_angles = None
    return cg_traj


def step1_save_raw(
    pre: Preprocessor,
    pdbid: str,
    cg_traj: mdtraj.Trajectory,
    cg_map: object,
    forces: np.ndarray,
    tick: Callable[[str], None],
) -> None:
    tick("Saving Data")
    pre.save_data(pdbid, cg_traj, cg_map.embeddings, forces)


def step1_attach_coords(mol: object, cg_traj: mdtraj.Trajectory) -> None:
    assert cg_traj.xyz is not None
    mol.coords = np.moveaxis(cg_traj.xyz, 0, -1)


def step1_prior_cache(
    pre: Preprocessor, pdbid: str, mol: object, cg_traj: mdtraj.Trajectory, tick: Callable[[str], None]
) -> None:
    tick("Generating prior fit data")
    if pre.prior_file:
        return
    fit_dir = pre.paths.pdb_fit(pdbid)
    fit_dir.mkdir(parents=True, exist_ok=True)
    pre.prior_builder.add_molecule(mol, cg_traj, os.fspath(fit_dir))


# --- Step 3 helpers ---


def step3_remove_stale_prior_files(pre: Preprocessor, pdbid: str) -> None:
    raw = pre.paths.pdb_raw(pdbid)
    for name in (f"{pdbid}_priors.yaml", f"{pdbid}_prior_params.json"):
        p = raw / name
        if p.exists():
            p.unlink()


def step3_delta_paths(pre: Preprocessor, pdbid: str) -> dict[str, str | None]:
    raw = pre.paths.pdb_raw(pdbid)
    box_npz = os.fspath(raw / "box.npy") if pre.box else None
    return {
        "coords": os.fspath(raw / "coordinates.npy"),
        "forces": os.fspath(raw / "forces.npy"),
        "delta": os.fspath(raw / "deltaforces.npy"),
        "prior_e": os.fspath(raw / "prior_energy.npy"),
        "box": box_npz,
        "ff": os.fspath(pre.paths.priors_yaml()),
        "psf": os.fspath(pre.paths.pdb_processed(pdbid) / f"{pdbid}_processed.psf"),
    }


def _step3_nn_and_classical(
    pre: Preprocessor,
    dfo: DeltaForces,
    paths: dict[str, str | None],
    bar_position: int,
    classical_workers: int,
) -> None:
    pp = pre.prior_builder.prior_params
    dfo.addExternalForces(
        paths["ff"],
        pre.prior_builder.priors["bonds"],
        pre.prior_builder.priors["angles"],
        pre.prior_builder.priors["dihedrals"],
        forceterms=pp["forceterms_nn"],
        bar_position=bar_position,
    )
    dfo.computePriorForces(
        paths["ff"],
        exclusions=pp["exclusions"],
        forceterms=pp["forceterms_classical"],
        bar_position=bar_position,
        num_parallel_workers=classical_workers,
    )


def step3_run_delta_forces(
    pre: Preprocessor, paths: dict[str, str | None], bar_position: int
) -> None:
    dfo = DeltaForces(
        pre.settings.device_step_3,
        paths["psf"],
        paths["coords"],
        paths["box"],
    )
    n_frames = int(dfo.coords.shape[0])
    classical_workers = pre._step3_classical_workers(n_frames)
    pp = pre.prior_builder.prior_params

    if "external" in pp.keys():
        _step3_nn_and_classical(pre, dfo, paths, bar_position, classical_workers)
    else:
        dfo.computePriorForces(
            paths["ff"],
            exclusions=pp["exclusions"],
            forceterms=pp["forceterms"],
            bar_position=bar_position,
            num_parallel_workers=classical_workers,
        )

    dfo.makeAndSaveDeltaForces(paths["forces"], paths["delta"], paths["prior_e"])


def _worker_bar_position(default: int = 1) -> int:
    name = mp.current_process().name
    parts = name.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1]) + 1
    return default
