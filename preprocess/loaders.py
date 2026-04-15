"""Trajectory discovery and HDF5 slice loading."""

from __future__ import annotations

import os

import h5py
import mdtraj

from .trajectory_source import H5BatchTrajectorySource


def slice_to_str(s: slice) -> str:
    result = [s.start, s.stop, s.step]
    result = [str(i) if i is not None else "" for i in result]
    return ":".join(result)


def get_prior_params_path(prior_path: str) -> str:
    dir_path, file_name = os.path.split(prior_path)
    file_name = file_name.replace("priors.yaml", "prior_params.json")
    return os.path.join(dir_path, file_name)


def load_h5_traj_slice(path: str, slice_: slice):
    """Load a slice from a h5 trajectory without reading the entire file into memory."""
    base_traj = mdtraj.load_frame(path, 0)
    with h5py.File(path) as f:
        t_xyz = f["coordinates"][slice_][:]  # pyright: ignore[reportIndexIssue]
        t_time = f["time"][slice_][:]  # pyright: ignore[reportIndexIssue]

        t_unitcell_lengths = None
        t_unitcell_angles = None
        if "cell_lengths" in f.keys():
            t_unitcell_lengths = f["cell_lengths"][slice_][:]  # pyright: ignore[reportIndexIssue]
            t_unitcell_angles = f["cell_angles"][slice_][:]  # pyright: ignore[reportIndexIssue]

    return mdtraj.Trajectory(
        t_xyz,
        base_traj.topology,
        time=t_time,
        unitcell_lengths=t_unitcell_lengths,
        unitcell_angles=t_unitcell_angles,
    )


def gen_input_mapping(conf: list) -> dict:
    """Find the list of input files for the passed dataset config."""
    pdbid_mapping: dict = {}
    for entry in conf:
        input_path = entry["path"]
        prefix = entry.get("prefix", "")
        suffix = entry.get("suffix", "")
        assert os.path.isdir(input_path), f"Input path does not exist: {input_path}"
        if "pdbids" in entry:
            for dir_name in entry["pdbids"]:
                input_h5 = os.path.join(input_path, dir_name, "result", f"output_{dir_name}.h5")
                assert os.path.exists(input_h5), "Requested path {input_path}/{dir_name} does not exist"
                pdbid_mapping[prefix + dir_name + suffix] = input_h5
        else:
            dir_names = os.listdir(input_path)
            for dir_name in sorted(dir_names):
                input_h5 = os.path.join(input_path, dir_name, "result", f"output_{dir_name}.h5")
                if os.path.exists(input_h5):
                    pdbid_mapping[prefix + dir_name + suffix] = input_h5
                else:
                    print(f'  Skipping "{dir_name}" (directory contains no output)')
    return pdbid_mapping


class BatchGeneratorH5Loader:
    """`TrajectorySource` built from the same YAML/directory discovery as `gen_input_mapping`."""

    def __init__(self, dataset_conf: list):
        self._source = H5BatchTrajectorySource(gen_input_mapping(dataset_conf))

    @property
    def source(self) -> H5BatchTrajectorySource:
        return self._source
