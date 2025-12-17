#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import yaml
import glob
import shutil
import pickle
import argparse
import traceback
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import h5py
import mdtraj
import torch
from tqdm import tqdm

# Your project imports
from module import prior, prior_flex, psfwriter
from module.make_deltaforces import DeltaForces
from module.cg_mapping import CGMapping
from module.torchmd_cg_mappings import CACB_MAP

torch.multiprocessing.set_sharing_strategy("file_system")


# -----------------------------
# MPI (optional)
# -----------------------------
try:
    from mpi4py import MPI  # type: ignore
    MPI_AVAILABLE = True
except Exception:
    MPI_AVAILABLE = False
    MPI = None  # type: ignore


# -----------------------------
# Small utilities (pure funcs)
# -----------------------------
def slice_to_str(s: slice) -> str:
    parts = [s.start, s.stop, s.step]
    return ":".join("" if p is None else str(p) for p in parts)

def parse_slice(s: str) -> slice:
    # "start:stop:step" with empty parts allowed
    parts = [p.strip() for p in s.split(":")]
    parts += [""] * (3 - len(parts))
    vals = [int(p) if p != "" else None for p in parts[:3]]
    return slice(*vals)

def get_prior_params_path(prior_yaml_path: str) -> str:
    d, fn = os.path.split(prior_yaml_path)
    return os.path.join(d, fn.replace("priors.yaml", "prior_params.json"))

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_h5_traj_slice(path: str, s: slice) -> Tuple[mdtraj.Trajectory, Optional[np.ndarray]]:
    base_traj = mdtraj.load_frame(path, 0)
    weights = None
    with h5py.File(path) as f:
        xyz = f["coordinates"][s][:]
        time = f["time"][s][:]
        unitcell_lengths = unitcell_angles = None
        if "cell_lengths" in f:
            unitcell_lengths = f["cell_lengths"][s][:]
            unitcell_angles = f["cell_angles"][s][:]
        if "weight" in f:
            weights = f["weight"][s][:]
    traj = mdtraj.Trajectory(
        xyz,
        base_traj.topology,
        time=time,
        unitcell_lengths=unitcell_lengths,
        unitcell_angles=unitcell_angles,
    )
    return traj, weights


# -----------------------------
# Config objects
# -----------------------------
@dataclass(frozen=True)
class RuntimeFlags:
    filter_not_processed_step_one: bool = False
    do_step_1: bool = True
    regen_cache_files: bool = True
    device_step_3: str = "cpu"
    use_cached_fits: Tuple[str, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class DistributedConfig:
    # If MPI is available and user passes --mpi, we will split pdbids by rank
    use_mpi: bool = False
    # If using MPI, you can still have per-rank multiprocessing
    per_rank_workers: int = 1

@dataclass
class PreprocessConfig:
    output_dir: str
    temp: int = 300
    frame_slice: slice = slice(None)
    optimize_forces: bool = False
    use_box: bool = True
    prior_plots: bool = True
    resume: bool = False
    fit_constraints: bool = True
    fit_min_cnt: int = 0

    prior_name: str = ""
    prior_file: Optional[str] = None


# -----------------------------
# Mapping definitions (kept minimal)
# -----------------------------
class CGMappingDef_CA:
    def __init__(self):
        residues = ["ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU","MET","ASN","PRO","HYP","GLN","ARG","SER","THR","VAL","TRP","TYR"]
        embedding_residues = ["ALA","ARG","ASN","ASP","ASX","CYS","GLU","GLN","GLX","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
        self.bead_embeddings = {name: [index + 1] for index, name in enumerate(sorted(embedding_residues))}
        self.bead_atom_selection = {k: [["CA"]] for k in residues}
        self.bead_types = {
            "ALA":["CAA"],"ARG":["CAR"],"ASN":["CAN"],"ASP":["CAD"],"CYS":["CAC"],"GLN":["CAQ"],"GLU":["CAE"],
            "GLY":["CAG"],"HIS":["CAH"],"HSD":["CAH"],"ILE":["CAI"],"LEU":["CAL"],"LYS":["CAK"],"MET":["CAM"],
            "PHE":["CAF"],"PRO":["CAP"],"SER":["CAS"],"THR":["CAT"],"TRP":["CAW"],"TYR":["CAY"],"VAL":["CAV"],
        }
        self.bead_atom_names = {k: ["CA"] for k in residues}
        self.bead_masses = {k: [12.01] for k in residues}
        self.bead_backbone_idx = {k: 0 for k in residues}

class CGMappingDef_CACB:
    def __init__(self):
        residues = ["ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU","MET","ASN","PRO","HYP","GLN","ARG","SER","THR","VAL","TRP","TYR"]
        self.bead_atom_selection = {k: [["CA"], ["CB"]] for k in residues}
        self.bead_atom_selection["GLY"] = [["CA"]]
        self.bead_types = {
            "ALA":["CA","CBA"],"ARG":["CA","CBR"],"ASN":["CA","CBN"],"ASP":["CA","CBD"],"CYS":["CA","CBC"],
            "GLN":["CA","CBQ"],"GLU":["CA","CBE"],"GLY":["CAG"],"HIS":["CA","CBH"],"HSD":["CA","CBH"],
            "ILE":["CA","CBI"],"LEU":["CA","CBL"],"LYS":["CA","CBK"],"MET":["CA","CBM"],"PHE":["CA","CBF"],
            "PRO":["CA","CBP"],"SER":["CA","CBS"],"THR":["CA","CBT"],"TRP":["CA","CBW"],"TYR":["CA","CBY"],
            "VAL":["CA","CBV"],
        }
        embedding_map = {k: i for i, k in enumerate(sorted(set.union(*[set(v) for v in self.bead_types.values()])))}
        self.bead_embeddings = {k: [embedding_map[i] for i in v] for k, v in self.bead_types.items()}
        self.bead_atom_names = {k: ["CA","CB"] for k in residues}
        self.bead_atom_names["GLY"] = ["CA"]
        self.bead_masses = {k: [12.01] * len(v) for k, v in self.bead_types.items()}
        self.bead_backbone_idx = {k: 0 for k in residues}


# -----------------------------
# PriorBuilder + PriorFactory
# -----------------------------
class PriorBuilder:
    """
    Base prior builder. Concrete behavior comes from:
      - mapping_def (CA vs CACB)
      - a list of term factories (bonds/angles/dihedrals/lj etc)
      - prior_params metadata (for downstream delta-forces logic)
    """
    def __init__(self, *, mapping_def: Any, prior_params: Dict[str, Any], term_factories: Dict[str, Callable[[], Any]], runtime: RuntimeFlags):
        self.mapping_def = mapping_def
        self.prior_params: Dict[str, Any] = dict(prior_params)
        self.term_factories = dict(term_factories)
        self.runtime = runtime

        self.terms: Dict[str, Any] = {k: f() for k, f in self.term_factories.items()}
        self.atom_types: set[str] = set()
        self.priors: Optional[Dict[str, Any]] = None

    # mapping
    def build_mapping(self, topology) -> CGMapping:
        return CGMapping(topology, self.mapping_def)

    def make_mol(self, cg_map: CGMapping):
        bonds = "bonds" in self.terms
        angles = "angles" in self.terms
        dihedrals = "dihedrals" in self.terms
        return cg_map.to_mol(bonds=bonds, angles=angles, dihedrals=dihedrals)

    # cache aggregation
    def add_molecule(self, mol, traj, cache_dir: str) -> None:
        fit_ok_path = os.path.join(cache_dir, "fit_ok.txt")
        if os.path.exists(fit_ok_path):
            os.unlink(fit_ok_path)

        for term in self.terms.values():
            term.add_molecule(mol, traj, cache_dir)
        self.atom_types |= set(mol.atomtype)

        np.save(os.path.join(cache_dir, "atomtype.npy"), mol.atomtype)
        with open(fit_ok_path, "wt", encoding="utf-8") as f:
            f.write("ok")

    def load_molecule_cache(self, cache_dir: str) -> None:
        assert os.path.exists(os.path.join(cache_dir, "fit_ok.txt"))
        atomtype = np.load(os.path.join(cache_dir, "atomtype.npy"), allow_pickle=True)
        self.atom_types |= set(atomtype)
        for term in self.terms.values():
            term.load_molecule_cache(cache_dir)

    # fitting
    def _init_prior_dict(self) -> None:
        priors: Dict[str, Any] = {}
        priors["atomtypes"] = sorted(self.atom_types)
        priors["bonds"] = {}
        priors["angles"] = {}
        priors["dihedrals"] = {}
        priors["lj"] = {}
        priors["electrostatics"] = {at: {"charge": 0.0} for at in priors["atomtypes"]}
        priors["masses"] = {at: 12.01 for at in priors["atomtypes"]}
        self.priors = priors

    def fit(self, temperature: int, plot_dir: Optional[str]) -> None:
        self._init_prior_dict()
        assert self.priors is not None

        for key, term in self.terms.items():
            cached = plot_dir and os.path.exists(f"{plot_dir}/prior_{key}.pkl") and (key in self.runtime.use_cached_fits)
            if cached:
                with open(f"{plot_dir}/prior_{key}.pkl", "rb") as f:
                    self.priors[key] = pickle.load(f)
                continue

            self.priors[key] = term.get_param(
                temperature,
                plot_dir,
                self.prior_params.get("fit_constraints", True),
                self.prior_params.get("min_cnt", 0),
            )
            if plot_dir:
                with open(f"{plot_dir}/prior_{key}.pkl", "wb") as f:
                    pickle.dump(self.priors[key], f)

    def save_prior(self, output_dir: str) -> None:
        assert self.priors is not None
        with open(os.path.join(output_dir, "priors.yaml"), "w") as f:
            yaml.dump(self.priors, f)
        with open(os.path.join(output_dir, "prior_params.json"), "w") as f:
            json.dump(self.prior_params, f, indent=2, sort_keys=True)

    # flex special-case: override by composition instead of subclass if you want
    def save_prior_flex(self, output_dir: str) -> None:
        """
        Mirrors your flex behavior: yaml gets only classical priors, pickle saves nets.
        """
        assert self.priors is not None

        # Write params
        with open(os.path.join(output_dir, "prior_params.json"), "w") as f:
            json.dump(self.prior_params, f, indent=2, sort_keys=True)

        # Truncate bonds/angles/dihedrals from YAML (classical only)
        truncated = dict(self.priors)
        truncated.pop("bonds", None)
        truncated.pop("angles", None)
        truncated.pop("dihedrals", None)

        with open(os.path.join(output_dir, "priors.yaml"), "w") as f:
            yaml.dump(truncated, f)

        payload = dict(self.priors)
        payload["terms"] = self.terms
        payload["prior_params"] = self.prior_params
        with open(os.path.join(output_dir, "priors.pkl"), "wb") as f:
            pickle.dump(payload, f)


class PriorFactory:
    """
    Defines priors declaratively (no giant subclass tree).
    """
    @staticmethod
    def make(prior_name: str, cfg: PreprocessConfig, runtime: RuntimeFlags) -> PriorBuilder:
        # Shared term constructors
        def bonds(): return prior.ParamBondedCalculator()
        def bonds_null(): return prior.NullParamBondedCalculator()
        def angles(center=False): return prior.ParamAngleCalculator(center=center)
        def angles_null(): return prior.NullParamAngleCalculator()
        def dihedrals(unified=False, scale=1.0): return prior.ParamDihedralCalculator(unified=unified, scale=scale)
        def dihedrals_null(): return prior.NullParamDihedralCalculator()
        def lj(exclusions=None): return prior.ParamNonbondedCalculator(fit_range=[3, 6], exclusion_terms=exclusions or set())

        def flex_bonds(): return prior_flex.ParamBondedFlexCalculator()
        def flex_angles(center=False): return prior_flex.ParamAngleFlexCalculator(center=center)
        def flex_dihedrals(unified=False): return prior_flex.ParamDihedralFlexCalculator(unified=unified)

        # Declarative catalog
        CATALOG: Dict[str, Dict[str, Any]] = {
            # CA
            "CA": {
                "mapping_def": CGMappingDef_CA(),
                "prior_params": {"prior_configuration_name":"CA","exclusions":["bonds"],"forceterms":["bonds"]},
                "terms": {"bonds": bonds},
            },
            "CA_lj": {
                "mapping_def": CGMappingDef_CA(),
                "prior_params": {"prior_configuration_name":"CA_lj","exclusions":["bonds"],"forceterms":["bonds","repulsioncg"]},
                "terms": {"bonds": bonds, "lj": lj},
            },
            "CA_lj_angleXCX_dihedralX": {
                "mapping_def": CGMappingDef_CA(),
                "prior_params": {"prior_configuration_name":"CA_lj_angleXCX_dihedralX","exclusions":["bonds","angles","dihedrals"],"forceterms":["bonds","angles","dihedrals","repulsioncg"]},
                "terms": {"bonds": bonds, "lj": lj, "angles": (lambda: angles(center=True)), "dihedrals": (lambda: dihedrals(unified=True))},
            },
            "CA_lj_bondNull_angleNull_dihedralNull": {
                "mapping_def": CGMappingDef_CA(),
                "prior_params": {"prior_configuration_name":"CA_lj_bondNull_angleNull_dihedralNull","exclusions":["bonds","angles","1-4"],"forceterms":["Bonds","angles","dihedrals","RepulsionCG"]},
                "terms": {"bonds": bonds_null, "lj": lj, "angles": angles_null, "dihedrals": dihedrals_null},
            },
            "CA_Majewski2022_v1": {
                "mapping_def": CGMappingDef_CA(),
                "prior_params": {"prior_configuration_name":"CA_Majewski2022_v1","exclusions":["bonds"],"forceterms":["bonds","dihedrals","repulsioncg"]},
                "terms": {"bonds": bonds, "lj": (lambda: lj(exclusions={"bonds"})), "dihedrals": (lambda: dihedrals(unified=True, scale=0.5))},
            },
            "CA_null": {
                "mapping_def": CGMappingDef_CA(),
                "prior_params": {"prior_configuration_name":"CA_null","exclusions":[],"forceterms":[]},
                "terms": {},
            },

            # CACB
            "CACB": {
                "mapping_def": CGMappingDef_CACB(),
                "prior_params": {"prior_configuration_name":"CACB","exclusions":["bonds"],"forceterms":["bonds"]},
                "terms": {"bonds": bonds},
            },
            "CACB_lj_angle_dihedral": {
                "mapping_def": CGMappingDef_CACB(),
                "prior_params": {"prior_configuration_name":"CACB_lj_angle_dihedral","exclusions":["bonds","angles","dihedrals"],"forceterms":["bonds","angles","dihedrals","repulsioncg"]},
                "terms": {"bonds": bonds, "lj": lj, "angles": angles, "dihedrals": dihedrals},
            },

            # FLEX
            "CA_lj_angleXCX_dihedralX_flex": {
                "mapping_def": CGMappingDef_CA(),
                "prior_params": {
                    "prior_configuration_name":"CA_lj_angleXCX_dihedralX_flex",
                    "exclusions":["bonds","angles","dihedrals"],
                    "forceterms_nn":["bonds","angles","dihedrals"],
                    "forceterms_classical":["repulsioncg"],
                    "external": True,
                },
                "terms": {
                    "bonds": flex_bonds,
                    "angles": (lambda: flex_angles(center=True)),
                    "dihedrals": (lambda: flex_dihedrals(unified=True)),
                    "lj": lj,
                },
            },
        }

        if prior_name not in CATALOG:
            raise RuntimeError(f"Unknown prior configuration: {prior_name}")

        spec = CATALOG[prior_name]

        # inject fit settings
        prior_params = dict(spec["prior_params"])
        prior_params["fit_constraints"] = cfg.fit_constraints
        prior_params["min_cnt"] = cfg.fit_min_cnt

        # flex convenience: build forceterms union
        if prior_params.get("external"):
            prior_params["forceterms"] = prior_params["forceterms_classical"] + prior_params["forceterms_nn"]

        return PriorBuilder(
            mapping_def=spec["mapping_def"],
            prior_params=prior_params,
            term_factories=spec["terms"],
            runtime=runtime,
        )


# -----------------------------
# Execution backends
# -----------------------------
class ExecContext:
    def __init__(self, dist: DistributedConfig):
        self.dist = dist
        self.rank = 0
        self.world = 1
        self.comm = None

        if dist.use_mpi:
            if not MPI_AVAILABLE:
                raise RuntimeError("MPI requested but mpi4py is not installed.")
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.world = self.comm.Get_size()

    def barrier(self) -> None:
        if self.comm:
            self.comm.Barrier()

    def is_root(self) -> bool:
        return self.rank == 0

    def split_items(self, items: List[str]) -> List[str]:
        # deterministic split by rank (round-robin keeps load more even with variable protein sizes)
        if self.world == 1:
            return items
        return [x for i, x in enumerate(items) if (i % self.world) == self.rank]


class ParallelRunner:
    """
    One DRY runner for:
      - serial
      - multiprocessing per rank
    """
    def __init__(self, workers: int):
        self.workers = max(1, int(workers))

    @staticmethod
    def _proc_init(counter: mp.Value) -> None:
        with counter.get_lock():
            idx = int(counter.value)
            counter.value += 1
        mp.current_process().name = f"PreprocessWorker-{idx}"

    @staticmethod
    def worker_bar_position(default: int = 1) -> int:
        try:
            return int(mp.current_process().name.split("-")[1]) + 1
        except Exception:
            return default

    def run(
        self,
        items: List[str],
        fn: Callable[[str], None],
        *,
        desc: str,
        show_progress: bool = True,
    ) -> Dict[str, str]:
        if self.workers == 1:
            errors: Dict[str, str] = {}
            it = tqdm(items, desc=desc, dynamic_ncols=True) if show_progress else items
            for x in it:
                try:
                    fn(x)
                except Exception as e:
                    traceback.print_tb(e.__traceback__)
                    errors[x] = str(e)
            return errors

        # multiprocessing
        tqdm.get_lock()
        errors: Dict[str, str] = {}
        counter = mp.Value("i", 0, lock=True)

        with tqdm(total=len(items), desc=desc, dynamic_ncols=True, disable=not show_progress) as pbar:
            with mp.Pool(self.workers, initializer=self._proc_init, initargs=(counter,)) as pool:
                pending: Dict[mp.pool.ApplyResult, str] = {
                    pool.apply_async(fn, args=(x,)): x for x in items
                }
                while pending:
                    for res in list(pending.keys()):
                        if not res.ready():
                            continue
                        x = pending.pop(res)
                        try:
                            res.get()
                        except Exception as e:
                            traceback.print_tb(e.__traceback__)
                            errors[x] = str(e)
                    pbar.n = len(items) - len(pending)
                    pbar.refresh()
                    if pending:
                        next(iter(pending)).wait(1)
        return errors


# -----------------------------
# Dataset discovery
# -----------------------------
class DatasetIndex:
    def __init__(self, dataset_conf: List[Dict[str, Any]]):
        self.dataset_conf = dataset_conf

    def build_input_map(self) -> Dict[str, str]:
        pdbid_mapping: Dict[str, str] = {}
        for entry in self.dataset_conf:
            input_path = entry["path"]
            prefix = entry.get("prefix", "")
            suffix = entry.get("suffix", "")
            assert os.path.isdir(input_path), f"Input path does not exist: {input_path}"

            if "pdbids" in entry:
                for dir_name in entry["pdbids"]:
                    input_h5 = os.path.join(input_path, dir_name, "result", f"output_{dir_name}.h5")
                    assert os.path.exists(input_h5), f"Requested path missing: {input_h5}"
                    pdbid_mapping[prefix + dir_name + suffix] = input_h5
                continue

            for dir_name in sorted(os.listdir(input_path)):
                input_h5 = os.path.join(input_path, dir_name, "result", f"output_{dir_name}.h5")
                if os.path.exists(input_h5):
                    pdbid_mapping[prefix + dir_name + suffix] = input_h5
        return pdbid_mapping


# -----------------------------
# Core pipeline (OO, modular)
# -----------------------------
class Preprocessor:
    def __init__(
        self,
        cfg: PreprocessConfig,
        runtime: RuntimeFlags,
        dist: DistributedConfig,
        dataset_conf: List[Dict[str, Any]],
        input_map: Dict[str, str],
        prior_builder: PriorBuilder,
    ):
        self.cfg = cfg
        self.runtime = runtime
        self.exec = ExecContext(dist)
        self.runner = ParallelRunner(workers=dist.per_rank_workers)

        self.dataset_conf = dataset_conf
        self.input_map = dict(input_map)  # pdbid -> h5 path
        self.prior_builder = prior_builder

        # optional filter: only keep proteins that completed step 1
        if runtime.filter_not_processed_step_one:
            processed = {p.split("/")[-3] for p in glob.glob(os.path.join(cfg.output_dir, "*/fit/fit_ok.txt"))}
            self.input_map = {k: v for k, v in self.input_map.items() if k in processed}

    # ---------- high level ----------
    def run(self) -> None:
        ensure_dir(os.path.join(self.cfg.output_dir, "result"))
        all_pdbids = sorted(self.input_map.keys())

        # MPI split (each rank gets a subset)
        pdbids = self.exec.split_items(all_pdbids)

        # only root writes global info.json (avoids collisions)
        if self.exec.is_root():
            self._write_run_info(all_pdbids)

        self.exec.barrier()

        # Step 1 (per-rank)
        if self.runtime.do_step_1:
            errs = self.runner.run(
                pdbids,
                self._step1_one,
                desc=f"[rank {self.exec.rank}] Step 1",
                show_progress=True,
            )
            if errs and self.exec.is_root():
                print("Step 1 errors (sample):", dict(list(errs.items())[:10]))

        self.exec.barrier()

        # Step 2 (ONLY root fits priors once, then broadcast/filesystem share)
        if not self.cfg.prior_file:
            if self.exec.is_root():
                self._fit_prior_once(all_pdbids)
            self.exec.barrier()
        else:
            if self.exec.is_root():
                self._copy_prior_files()
            self.exec.barrier()

        # Step 3 (per-rank)
        errs3 = self.runner.run(
            pdbids,
            self._step3_one,
            desc=f"[rank {self.exec.rank}] Step 3",
            show_progress=True,
        )
        if errs3 and self.exec.is_root():
            print("Step 3 errors (sample):", dict(list(errs3.items())[:10]))

        self.exec.barrier()

        # root writes ok_list
        if self.exec.is_root():
            ok_list_path = os.path.join(self.cfg.output_dir, "result/ok_list.txt")
            with open(ok_list_path, "wt", encoding="utf-8") as f:
                f.write("\n".join(all_pdbids))
            print("Done!")

    # ---------- info / resume ----------
    def _write_run_info(self, all_pdbids: List[str]) -> None:
        info_path = os.path.join(self.cfg.output_dir, "result/info.json")
        payload = {
            "input_paths": [i["path"] for i in self.dataset_conf],
            "frame_slice": slice_to_str(self.cfg.frame_slice),
            "pdbids": all_pdbids,
            "optimize_forces": self.cfg.optimize_forces,
            "box": self.cfg.use_box,
            "prior_name": self.cfg.prior_name,
        }

        if self.cfg.resume and os.path.exists(info_path):
            with open(info_path, "rt", encoding="utf-8") as f:
                prev = json.load(f)
            for k in ["box", "frame_slice", "optimize_forces", "prior_name"]:
                if payload[k] != prev.get(k):
                    raise AssertionError(f"Can't resume with different parameters: {k}: {payload[k]} != {prev.get(k)}")

        with open(info_path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        # truncate ok_list
        with open(os.path.join(self.cfg.output_dir, "result/ok_list.txt"), "wt", encoding="utf-8") as _:
            pass

    # ---------- step 1 ----------
    def _step1_one(self, pdbid: str) -> None:
        cache_dir = os.path.join(self.cfg.output_dir, pdbid, "fit")
        if self.cfg.resume and os.path.exists(os.path.join(cache_dir, "fit_ok.txt")):
            return

        bar_pos = self.runner.worker_bar_position()
        self._process_step1(pdbid, bar_pos)

    def _save_raw(
        self,
        out_dir: str,
        *,
        embeddings: np.ndarray,
        forces: np.ndarray,
        coords: np.ndarray,
        box_vectors: Optional[np.ndarray],
        weights: Optional[np.ndarray],
    ) -> None:
        raw = os.path.join(out_dir, "raw")
        ensure_dir(raw)
        np.save(os.path.join(raw, "embeddings.npy"), embeddings)
        np.save(os.path.join(raw, "forces.npy"), forces)
        np.save(os.path.join(raw, "coordinates.npy"), coords)

        box_path = os.path.join(raw, "box.npy")
        if box_vectors is not None:
            np.save(box_path, box_vectors)
        elif os.path.exists(box_path):
            os.unlink(box_path)

        w_path = os.path.join(raw, "forces_weights.npy")
        if weights is not None:
            np.save(w_path, weights)
        elif os.path.exists(w_path):
            os.unlink(w_path)

    def _process_step1(self, pdbid: str, bar_position: int) -> None:
        input_h5 = self.input_map[pdbid]
        out_dir = os.path.join(self.cfg.output_dir, pdbid)
        ensure_dir(os.path.join(out_dir, "raw"))
        ensure_dir(os.path.join(out_dir, "processed"))

        with tqdm(total=7, position=bar_position, desc=f"{pdbid}: step1", dynamic_ncols=True, leave=False) as pbar:
            def step(msg: str) -> None:
                pbar.update(1)
                pbar.set_description_str(f"{pdbid}: {msg}")

            step("Loading trajectory")
            aa_traj, aa_weights = load_h5_traj_slice(input_h5, self.cfg.frame_slice)
            assert aa_traj.xyz is not None
            aa_traj.xyz *= 10.0  # nm -> Å

            step("Building CG mapping")
            cg_map = self.prior_builder.build_mapping(aa_traj.topology)
            mol = self.prior_builder.make_mol(cg_map)

            # optional debug topology save
            mol.write(os.path.join(out_dir, "processed", f"{pdbid}_processed.psf"))
            cg_topology_mol = cg_map.to_mol(bonds=True, angles=True, dihedrals=True)
            cg_topology_mol.write(os.path.join(out_dir, "processed", "topology.psf"))

            step("Mapping CG forces")
            with h5py.File(input_h5, "r") as f:
                forces = f["forces"][self.cfg.frame_slice, :, :]
            forces = cg_map.cg_optimal_forces(aa_traj, forces) if self.cfg.optimize_forces else cg_map.cg_forces(forces)
            forces = forces * 0.02390057361376673  # kJ/mol/nm -> kcal/mol/Å

            step("Mapping CG coordinates")
            xyz = cg_map.cg_positions(aa_traj.xyz)
            cg_traj = mdtraj.Trajectory(xyz, topology=cg_map.to_mdtraj())

            if self.cfg.use_box and aa_traj.unitcell_lengths is not None:
                cg_traj.unitcell_lengths = aa_traj.unitcell_lengths * 10.0
                cg_traj.unitcell_angles = aa_traj.unitcell_angles
            else:
                cg_traj.unitcell_lengths = None
                cg_traj.unitcell_angles = None

            cg_weights = None
            if aa_weights is not None:
                if aa_weights.ndim == 2:
                    cg_weights = cg_map.cg_weights(aa_weights)
                elif aa_weights.ndim == 1:
                    n_beads = len(cg_map.embeddings)
                    cg_weights = np.repeat(aa_weights[:, None], n_beads, axis=1)

            step("Saving raw arrays")
            self._save_raw(
                out_dir,
                embeddings=cg_map.embeddings,
                forces=forces,
                coords=cg_traj.xyz,
                box_vectors=(cg_traj.unitcell_vectors if self.cfg.use_box else None),
                weights=cg_weights,
            )

            step("Generating prior fit data")
            if not self.cfg.prior_file:
                cache_dir = os.path.join(out_dir, "fit")
                ensure_dir(cache_dir)
                mol.coords = np.moveaxis(cg_traj.xyz, 0, -1)
                self.prior_builder.add_molecule(mol, cg_traj, cache_dir)

    # ---------- step 2 ----------
    def _fit_prior_once(self, all_pdbids: List[str]) -> None:
        # cache prior_builder aggregation
        pb_cache = os.path.join(self.cfg.output_dir, "prior_builder.pkl")

        if os.path.exists(pb_cache) and not self.runtime.regen_cache_files:
            with open(pb_cache, "rb") as f:
                self.prior_builder = pickle.load(f)
        else:
            for pdbid in tqdm(all_pdbids, desc="Merging fit caches", dynamic_ncols=True):
                cache_dir = os.path.join(self.cfg.output_dir, pdbid, "fit")
                self.prior_builder.load_molecule_cache(cache_dir)
            with open(pb_cache, "wb") as f:
                pickle.dump(self.prior_builder, f)

        plot_dir = None
        if self.cfg.prior_plots:
            plot_dir = os.path.join(self.cfg.output_dir, "prior_fit_plots")
            ensure_dir(plot_dir)

        self.prior_builder.fit(self.cfg.temp, plot_dir=plot_dir)

        # Save priors
        if self.prior_builder.prior_params.get("external"):
            self.prior_builder.save_prior_flex(self.cfg.output_dir)
        else:
            self.prior_builder.save_prior(self.cfg.output_dir)

    def _copy_prior_files(self) -> None:
        assert self.cfg.prior_file is not None
        prior_params_path = get_prior_params_path(self.cfg.prior_file)
        shutil.copy(self.cfg.prior_file, os.path.join(self.cfg.output_dir, "priors.yaml"))
        shutil.copy(prior_params_path, os.path.join(self.cfg.output_dir, "prior_params.json"))

    # ---------- step 3 ----------
    def _step3_one(self, pdbid: str) -> None:
        bar_pos = self.runner.worker_bar_position()
        self._process_step3(pdbid, bar_pos)

    def _process_step3(self, pdbid: str, bar_position: int) -> None:
        out_dir = os.path.join(self.cfg.output_dir, pdbid)
        raw_dir = os.path.join(out_dir, "raw")
        proc_dir = os.path.join(out_dir, "processed")

        # legacy cleanup
        for legacy in [f"{pdbid}_priors.yaml", f"{pdbid}_prior_params.json"]:
            p = os.path.join(raw_dir, legacy)
            if os.path.exists(p):
                os.unlink(p)

        coords_npz = os.path.join(raw_dir, "coordinates.npy")
        forces_npz = os.path.join(raw_dir, "forces.npy")
        delta_forces_npz = os.path.join(raw_dir, "deltaforces.npy")
        prior_energy_npz = os.path.join(raw_dir, "prior_energy.npy")
        box_npz = os.path.join(raw_dir, "box.npy") if self.cfg.use_box else None

        forcefield = os.path.join(self.cfg.output_dir, "priors.yaml")
        psf_file = os.path.join(proc_dir, f"{pdbid}_processed.psf")

        prior_params = self.prior_builder.prior_params
        df = DeltaForces(self.runtime.device_step_3, psf_file, coords_npz, box_npz)

        if prior_params.get("external"):
            df.addExternalForces(
                forcefield,
                self.prior_builder.priors["bonds"],
                self.prior_builder.priors["angles"],
                self.prior_builder.priors["dihedrals"],
                forceterms=prior_params["forceterms_nn"],
                bar_position=bar_position,
            )
            df.computePriorForces(
                forcefield,
                exclusions=prior_params["exclusions"],
                forceterms=prior_params["forceterms_classical"],
                bar_position=bar_position,
            )
        else:
            df.computePriorForces(
                forcefield,
                exclusions=prior_params["exclusions"],
                forceterms=prior_params["forceterms"],
                bar_position=bar_position,
            )

        df.makeAndSaveDeltaForces(forces_npz, delta_forces_npz, prior_energy_npz)


# -----------------------------
# CLI
# -----------------------------
def load_dataset_conf(inputs: List[str]) -> List[Dict[str, Any]]:
    conf: List[Dict[str, Any]] = []
    for i in inputs:
        if os.path.isfile(i):
            with open(i, "r") as f:
                conf += yaml.safe_load(f)
        else:
            conf += [{"path": i}]
    return conf

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess data (OO, DRY, MPI-enabled).")
    parser.add_argument("input", nargs="+", help="Input directories or a YAML config file")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument("--pdbids", nargs="*", help="Specific PDB IDs to process")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames to process")
    parser.add_argument("--frame-slice", type=str, default=None, help="Frame slice: start:end:stride")
    parser.add_argument("--temp", type=int, default=300)
    parser.add_argument("--prior", type=str, default=None, help="Prior configuration name")
    parser.add_argument("--prior-file", default=None, help="Use a pre-fit priors.yaml (and matching prior_params.json)")
    parser.add_argument("--optimize-forces", action="store_true")
    parser.add_argument("--no-box", action="store_true")
    parser.add_argument("--prior-plots", dest="prior_plots", action="store_true", default=True)
    parser.add_argument("--no-prior-plots", dest="prior_plots", action="store_false")
    parser.add_argument("--no-fit-constraints", action="store_true")
    parser.add_argument("--fit-min-cnt", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-cached-fits", nargs="*", default=[], help="e.g. bonds angles dihedrals lj")

    # distributed
    parser.add_argument("--mpi", action="store_true", help="Enable MPI splitting (requires mpi4py, launch with mpirun)")
    parser.add_argument("--workers", type=int, default=32, help="Workers per rank (multiprocessing). Use 1 to disable.")

    # runtime flags (formerly globals)
    parser.add_argument("--filter-not-processed-step-one", action="store_true")
    parser.add_argument("--skip-step-1", action="store_true")
    parser.add_argument("--no-regen-cache-files", action="store_true")
    parser.add_argument("--device-step-3", type=str, default="cpu")

    args = parser.parse_args()
    print(args)

    assert not (args.num_frames and args.frame_slice)
    if args.num_frames is not None:
        frame_slice = slice(0, args.num_frames)
    elif args.frame_slice is not None:
        frame_slice = parse_slice(args.frame_slice)
    else:
        frame_slice = slice(None)

    cfg = PreprocessConfig(
        output_dir=args.output,
        temp=args.temp,
        frame_slice=frame_slice,
        optimize_forces=args.optimize_forces,
        use_box=not args.no_box,
        prior_plots=args.prior_plots,
        resume=args.resume,
        fit_constraints=not args.no_fit_constraints,
        fit_min_cnt=args.fit_min_cnt,
        prior_name=args.prior or "",
        prior_file=args.prior_file,
    )

    runtime = RuntimeFlags(
        filter_not_processed_step_one=args.filter_not_processed_step_one,
        do_step_1=not args.skip_step_1,
        regen_cache_files=not args.no_regen_cache_files,
        device_step_3=args.device_step_3,
        use_cached_fits=tuple(args.use_cached_fits),
    )

    dist = DistributedConfig(
        use_mpi=args.mpi,
        per_rank_workers=max(1, args.workers),
    )

    # If flex prior needs spawn, do it early
    # We'll detect it after we build the prior
    # (safe to call set_start_method once, guarded)
    torch.set_num_threads(1)

    # thread-safe matplotlib backend for plots
    import matplotlib
    matplotlib.use("Agg")

    # dataset
    dataset_conf = load_dataset_conf(args.input)
    index = DatasetIndex(dataset_conf)
    input_map = index.build_input_map()
    if args.pdbids:
        input_map = {p: input_map[p] for p in args.pdbids}

    # prior file sanity (if used)
    if cfg.prior_file:
        assert os.path.exists(cfg.prior_file), f"Prior file does not exist: {cfg.prior_file}"
        params_path = get_prior_params_path(cfg.prior_file)
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        if not cfg.prior_name:
            cfg.prior_name = params["prior_configuration_name"]
        elif cfg.prior_name != params["prior_configuration_name"]:
            print(f'WARNING: --prior "{cfg.prior_name}" differs from prior file "{params["prior_configuration_name"]}"')

    assert cfg.prior_name, "You must specify the prior via --prior or --prior-file"

    prior_builder = PriorFactory.make(cfg.prior_name, cfg, runtime)

    if prior_builder.prior_params.get("external"):
        # flex uses torch nets; spawn is safer in multiprocess contexts
        try:
            mp.set_start_method("spawn", force=True)
        except Exception:
            pass

    pre = Preprocessor(cfg, runtime, dist, dataset_conf, input_map, prior_builder)
    pre.run()


if __name__ == "__main__":
    main()

