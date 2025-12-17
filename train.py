#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import time
import yaml
import shutil
import itertools
import datetime
import traceback
import resource
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# Distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Local modules (unchanged)
from module.torchmdnet.model import create_model
from module import dataset
from module import model_util
from module.lr_scheduler_wrappers import (
    SchedulerWrapper_CosineAnnealingWarmRestarts,
    SchedulerWrapper_CosineAnnealingLR,
    SchedulerWrapper_ExponentialLR,
    SchedulerWrapper_ReduceLROnPlateau,
)

# ----------------------------- Small utilities -----------------------------

def flatten_first(t: Optional[Tensor]) -> Optional[Tensor]:
    """Flatten first two dims, preserving remaining dims."""
    if t is None or getattr(t, "shape", None) is None:
        return t
    if len(t.shape) < 2:
        return t
    return t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])

def deterministic_shuffle(items: List[str], seed: int) -> List[str]:
    """Deterministically shuffle a list."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n=len(items), generator=g, device="cpu")
    return [items[i] for i in idx]

def check_early_stopping(val_list: List[float], patience: int = 1) -> bool:
    """True if val loss increased (patience+1) consecutive times. patience<0 disables."""
    if patience < 0:
        return False
    if len(val_list) < patience + 2:
        return False
    window = np.array(val_list[-(patience + 2):])
    if np.all((window[1:] - window[:-1]) > 0):
        print(f"Validation loss increased {patience+1} times, stopping...")
        return True
    return False

def make_term_offsets(lengths: List[int], term_lengths: Tensor) -> Tensor:
    """Offset per-term indices across concatenated variable-length batches."""
    result = []
    count = 0
    repeats = len(term_lengths) // len(lengths)
    lengths = np.tile(np.array(lengths), repeats).tolist()
    assert len(lengths) == len(term_lengths)
    for off, nterms in zip(lengths, term_lengths):
        result.append(torch.full((int(nterms), 1), count, dtype=torch.long))
        count += int(off)
    return torch.cat(result, dim=0)

# ----------------------------- Domain config ------------------------------

class TermDef:
    def __init__(self, path: Optional[str] = None, conf: Optional[dict] = None):
        self.scales: Dict[str, float] = {}
        self.angle_wrap: Dict[str, bool] = {}
        if path:
            with open(path, "r") as f:
                conf = yaml.safe_load(f)
        if conf:
            for k, v in conf.items():
                self.scales[k] = float(v["scale"]) if (v is not None and "scale" in v) else 1.0
                self.angle_wrap[k] = bool(v["angle_wrap"]) if (v is not None and "angle_wrap" in v) else False

    def names(self) -> List[str]:
        return list(self.scales.keys())

    def scale(self, name: str) -> float:
        return self.scales[name]

    def wrap(self, name: str) -> bool:
        return self.angle_wrap[name]

# ----------------------------- Distributed manager -------------------------

@dataclass
class DistInfo:
    rank: Optional[int] = None
    world_size: Optional[int] = None
    local_rank: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self.rank is not None and self.world_size is not None and self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank is None or self.rank == 0

class DistributedManager:
    def __init__(self, enable: bool):
        self.info = DistInfo()
        self._requested = enable

    def setup(self) -> DistInfo:
        """Initialize distributed if env vars indicate multi-proc or user requested."""
        if not self._requested and not any(k in os.environ for k in ("RANK", "WORLD_SIZE", "SLURM_PROCID")):
            return self.info

        rank, world_size, local_rank = self._read_env_ranks()
        if rank is None:
            return self.info

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        self.info = DistInfo(rank=rank, world_size=world_size, local_rank=local_rank)
        print(f"Initialized process {rank}/{world_size} (local_rank={local_rank})")
        return self.info

    def cleanup(self) -> None:
        """Destroy process group if initialized."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def barrier(self) -> None:
        """Sync all processes if distributed."""
        if self.info.enabled:
            dist.barrier()

    def broadcast_bool(self, value: bool, src: int = 0, device: Optional[torch.device] = None) -> bool:
        """Broadcast a boolean from src to all processes."""
        if not self.info.enabled:
            return value
        assert device is not None
        t = torch.tensor(1 if value else 0, device=device)
        dist.broadcast(t, src=src)
        return bool(t.item())

    def all_reduce_sum(self, x: float, device: torch.device) -> float:
        """All-reduce a scalar float via SUM."""
        if not self.info.enabled:
            return x
        t = torch.tensor(x, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    def _read_env_ranks(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Read rank/world_size/local_rank from torchrun or SLURM env vars."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return rank, world_size, local_rank

        if "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            local_rank = int(os.environ.get("SLURM_LOCALID", 0))
            return rank, world_size, local_rank

        return None, None, None

# ----------------------------- Model wrappers ------------------------------

class BatchWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pos: Tensor, lengths: List[int], **kwargs) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Prepare batch indices/terms and forward into model."""
        batch_nums = dataset.make_batch_nums(len(pos), lengths).to(pos.device)
        kwargs["pos"] = pos

        for k, v in list(kwargs.items()):
            kwargs[k] = flatten_first(v)

        if "bonds" in kwargs:
            kwargs["bonds"] = kwargs["bonds"] + make_term_offsets(lengths, kwargs.pop("len_bonds").cpu()).to(pos.device)
        if "angles" in kwargs:
            kwargs["angles"] = kwargs["angles"] + make_term_offsets(lengths, kwargs.pop("len_angles").cpu()).to(pos.device)
        if "dihedrals" in kwargs:
            kwargs["dihedrals"] = kwargs["dihedrals"] + make_term_offsets(lengths, kwargs.pop("len_dihedrals").cpu()).to(pos.device)

        kwargs["batch"] = batch_nums
        result = self.model(**kwargs)

        if len(result) == 2:
            out_e, out_f = result
            return out_e, out_f, {}
        out_e, out_f, extra = result
        return out_e, out_f, extra

class ModelManager:
    def __init__(self, conf: Dict[str, Any], dist_info: DistInfo, gpu_ids, local_rank: Optional[int]):
        self.conf = conf
        self.dist = dist_info
        self.gpu_ids = gpu_ids
        self.local_rank = local_rank

        self.device = self._select_device()
        self.model = create_model(args=conf).to(self.device)
        self.wrapped = BatchWrapper(self.model)
        self.parallel, self.device_output = self._wrap_parallel()

    def _select_device(self) -> torch.device:
        """Pick a torch device for the current process."""
        if self.local_rank is not None:
            return torch.device(f"cuda:{self.local_rank}")
        if self.gpu_ids != "cpu":
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _wrap_parallel(self) -> Tuple[nn.Module, Any]:
        """Wrap model in DDP or DataParallel or no-wrap."""
        if self.dist.enabled:
            parallel = DDP(
                self.wrapped,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
            if self.dist.is_main:
                print(f"DDP: Training on {self.dist.world_size} GPUs across nodes (local_rank={self.local_rank})")
            return parallel, self.device

        if self.gpu_ids == "cpu":
            if self.dist.is_main:
                print("Training on CPU")
            return self.wrapped, "cpu"

        parallel = nn.DataParallel(self.wrapped, device_ids=self.gpu_ids)
        if self.dist.is_main:
            print(f"DataParallel: Training on {len(parallel.device_ids)} GPU(s)")
        return parallel, parallel.output_device

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.parallel.eval()

    def state_target_for_loading(self) -> nn.Module:
        """Return the object to pass to model_util.load_state_dict_with_rename."""
        if isinstance(self.parallel, DDP):
            return self.parallel.module.model
        if hasattr(self.parallel, "module"):
            return self.parallel.module.model
        return self.model

    def state_dict_for_saving(self) -> Dict[str, Tensor]:
        """Extract the underlying model state dict in a wrapper-agnostic way."""
        if isinstance(self.parallel, DDP):
            return self.parallel.module.model.state_dict()
        if hasattr(self.parallel, "module"):
            return self.parallel.module.model.state_dict()
        return self.model.state_dict()

# ----------------------------- Data module ---------------------------------

class RoundRobinDataWrapper:
    def __init__(self, *iterables: Iterable):
        self.iterables = iterables

    def __len__(self) -> int:
        return sum(map(len, self.iterables))

    def __iter__(self):
        iters = map(iter, self.iterables)
        for num_active in range(len(self.iterables), 0, -1):
            iters = itertools.cycle(itertools.islice(iters, num_active))
            yield from map(next, iters)

@dataclass
class DataBundle:
    datasets: List[Any]
    train: RoundRobinDataWrapper
    val: RoundRobinDataWrapper
    train_samplers: List[Optional[DistributedSampler]]
    pdb_list: List[str]

class DataModule:
    def __init__(
        self,
        directory_path: str,
        subsetpdbs: str,
        val_ratio: float,
        batch_size: int,
        atoms_per_call: Optional[int],
        enable_shuffle: bool,
        dataset_chunk_size: Optional[int],
        use_npfile: bool,
        embedding_filename: Optional[str],
        energy_matching: bool,
        dist_info: DistInfo,
    ):
        self.dir = directory_path
        self.subsetpdbs = subsetpdbs
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.atoms_per_call = atoms_per_call
        self.enable_shuffle = enable_shuffle
        self.chunk = dataset_chunk_size
        self.use_npfile = use_npfile
        self.embedding_filename = embedding_filename or "embeddings.npy"
        self.energy_filename = "tica_delta_energies.npy" if energy_matching else None
        self.dist = dist_info

    def build(self) -> DataBundle:
        """Create datasets and round-robin dataloaders (optionally chunked)."""
        pdb_list = self._load_pdb_list()
        pdb_lists = [pdb_list[i:i + self.chunk] for i in range(0, len(pdb_list), self.chunk)] if self.chunk else [pdb_list]

        datasets, train_loaders, val_loaders, samplers = [], [], [], []
        for chunk_list in pdb_lists:
            ds, tr, va, sampler = self._make_loaders(chunk_list)
            datasets.append(ds)
            train_loaders.append(tr)
            val_loaders.append(va)
            samplers.append(sampler)

        return DataBundle(
            datasets=datasets,
            train=RoundRobinDataWrapper(*train_loaders),
            val=RoundRobinDataWrapper(*val_loaders),
            train_samplers=samplers,
            pdb_list=pdb_list,
        )

    def _load_pdb_list(self) -> List[str]:
        """Read, dedupe, sort, and deterministically shuffle PDB IDs."""
        with open(os.path.join(self.dir, "result", self.subsetpdbs), "r") as f:
            pdb_list = [x for x in f.read().split("\n") if x]
        pdb_list = sorted(set(pdb_list))
        return deterministic_shuffle(pdb_list, seed=47563537)

    def _make_loaders(self, pdb_list: List[str]):
        """Build torch DataLoaders and (optionally) DistributedSamplers."""
        print("Dataset:", " ".join(pdb_list))
        all_data = dataset.ProteinDataset(
            self.dir,
            pdb_list,
            energy_file=self.energy_filename,
            embeddings_file=self.embedding_filename,
            use_npfile=self.use_npfile,
        )

        assert 0.0 < self.val_ratio < 1.0
        val_size = int(self.val_ratio * len(all_data))
        train_size = len(all_data) - val_size

        if self.enable_shuffle:
            g = torch.Generator().manual_seed(12341234)
            val_idx, train_idx = torch.utils.data.random_split(
                torch.arange(len(all_data)),
                [val_size, train_size],
                generator=g,
            )
        else:
            train_idx = range(train_size)
            val_idx = range(train_size, train_size + val_size)

        train = torch.utils.data.Subset(all_data, train_idx)
        val = torch.utils.data.Subset(all_data, val_idx)

        collate_fn = dataset.ProteinBatchCollate(self.atoms_per_call)

        train_sampler = None
        val_sampler = None
        if self.dist.enabled:
            train_sampler = DistributedSampler(train, num_replicas=self.dist.world_size, rank=self.dist.rank, shuffle=False)
            val_sampler = DistributedSampler(val, num_replicas=self.dist.world_size, rank=self.dist.rank, shuffle=False)

        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=False if train_sampler is None else False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=val_sampler,
        )
        return all_data, train_loader, val_loader, train_sampler

# ----------------------------- Checkpointing -------------------------------

class Checkpointer:
    def __init__(self, result_dir: str, dist_info: DistInfo):
        self.result_dir = result_dir
        self.dist = dist_info

    def find_resume_checkpoint(self) -> Optional[str]:
        """Pick mini checkpoint first, else normal checkpoint."""
        mini = os.path.join(self.result_dir, "checkpoint-mini.pth")
        main = os.path.join(self.result_dir, "checkpoint.pth")
        if os.path.exists(mini):
            return mini
        if os.path.exists(main):
            return main
        return None

    def save(
        self,
        path: str,
        epoch: int,
        model_state: Dict[str, Tensor],
        optimizer: optim.Optimizer,
        model_conf: Dict[str, Any],
        scheduler: Optional[Any],
        extra: Optional[dict] = None,
    ) -> None:
        """Save checkpoint only on main process."""
        if not self.dist.is_main:
            return
        ckpt = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "state_dict": model_state,
            "hyper_parameters": model_conf,
        }
        if scheduler:
            ckpt["scheduler"] = scheduler.state_dict()
        if extra:
            ckpt["extra"] = extra
        torch.save(ckpt, path)

    def write_training_info(
        self,
        epoch: int,
        directory_path: str,
        pdb_list: List[str],
        params: Dict[str, Any],
    ) -> None:
        """Write/update training_info.json and copy priors on main process."""
        if not self.dist.is_main:
            return

        training_info_path = os.path.join(self.result_dir, "training_info.json")
        info = {}

        if os.path.exists(training_info_path):
            with open(training_info_path, "r") as f:
                info = json.load(f)
            if "input_directory" in info:
                info = {"0": info}
        else:
            print("Path", training_info_path, "does not exist")

        info[str(epoch)] = {
            "weight_decay": params["weight_decay"],
            "learning_rate": params["learning_rate"],
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "input_directory": directory_path,
            "pdbs": pdb_list,
            "energy_weight": params["energy_weight"],
            "force_weight": params["force_weight"],
            "embedding_filename": params["embedding_filename"],
            "world_size": params.get("world_size", 1),
        }
        if params.get("lr_scheduler_repr") is not None:
            info[str(epoch)]["lr_scheduler"] = params["lr_scheduler_repr"]

        if not params["dry_run"]:
            with open(training_info_path, "w") as f:
                json.dump(info, f, indent=2)

        prior_path = os.path.join(directory_path, "priors.yaml")
        if os.path.exists(prior_path):
            prior_params_path = os.path.join(directory_path, "prior_params.json")
            shutil.copy(prior_path, self.result_dir)
            shutil.copy(prior_params_path, self.result_dir)

# ----------------------------- Training bookkeeping ------------------------

@dataclass
class History:
    train: List[float]
    val: List[float]
    energy: List[float]
    force: List[float]

    @classmethod
    def empty(cls) -> "History":
        return cls(train=[], val=[], energy=[], force=[])

    @classmethod
    def load(cls, result_dir: str) -> "History":
        path = os.path.join(result_dir, "history.npy")
        if not os.path.exists(path):
            return cls.empty()
        data = np.load(path, allow_pickle=True).item()
        return cls(train=data["train"], val=data["val"], energy=data["energy"], force=data["force"])

    def save(self, result_dir: str) -> None:
        np.save(os.path.join(result_dir, "history.npy"), {"train": self.train, "val": self.val, "energy": self.energy, "force": self.force})

class EpochHistory:
    def __init__(self, result_dir: str):
        self.result_dir = result_dir
        self.path = os.path.join(result_dir, "epoch_history.json")
        self.data: Dict[str, Any] = {}
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.data = json.load(f)

    def update(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

# ----------------------------- Optim & scheduler ---------------------------

def should_decay(param_name: str) -> bool:
    """Decide if a parameter should receive weight decay."""
    parts = param_name.split(".")
    assert parts
    if parts[-1] == "bias":
        return False
    if len(parts) >= 2 and parts[-2] == "embedding":
        return False
    if len(parts) >= 2 and parts[-2] == "distance_expansion":
        return False
    assert parts[-1] == "weight"
    return True

class OptimFactory:
    @staticmethod
    def adamw(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
        """Create AdamW with split decay groups."""
        do_decay, dont_decay = [], []
        for name, p in model.named_parameters():
            (do_decay if should_decay(name) else dont_decay).append(p)
        return optim.AdamW(
            [{"params": do_decay, "weight_decay": weight_decay}, {"params": dont_decay}],
            lr=lr,
        )

class SchedulerFactory:
    @staticmethod
    def from_args(args) -> Optional[Any]:
        """Build one of the scheduler wrappers from argparse args."""
        lr_scheduler = None
        if args.cos_anneal:
            T_0, T_mult = [int(i) for i in args.cos_anneal.split(",")]
            lr_scheduler = SchedulerWrapper_CosineAnnealingWarmRestarts(T_0, T_mult)
        if args.cos_lr:
            assert lr_scheduler is None
            T_max, eta_min = args.cos_lr.split(",")
            lr_scheduler = SchedulerWrapper_CosineAnnealingLR(int(T_max), float(eta_min))
        if args.exp_lr:
            assert lr_scheduler is None
            lr_scheduler = SchedulerWrapper_ExponentialLR(float(args.exp_lr))
        if args.plateau_lr:
            assert lr_scheduler is None
            factor, patience, threshold, min_lr = args.plateau_lr.split(",")
            lr_scheduler = SchedulerWrapper_ReduceLROnPlateau(float(factor), int(patience), float(threshold), float(min_lr))
        return lr_scheduler

# ----------------------------- Core trainer --------------------------------

@dataclass
class TrainArgs:
    directory_path: str
    result_directory: Optional[str]
    conf_path: str
    gpu_ids: Any
    weight_decay: float
    learning_rate: float
    epochs: int
    batch_size: int
    val_ratio: float
    atoms_per_call: Optional[int]
    scheduler: Optional[Any]
    dry_run: bool
    reset_early_stopping: bool
    enable_shuffle: bool
    mini_epoch_size: Optional[int]
    early_stopping: int
    checkpoint_save: int
    subsetpdbs: str
    energy_weight: float
    force_weight: float
    energy_matching: bool
    train_term_def: TermDef
    embedding_filename: Optional[str]
    dataset_chunk_size: Optional[int]
    use_npfile: bool
    use_force_weights: bool

class Trainer:
    def __init__(self, args: TrainArgs, dist_mgr: DistributedManager):
        self.args = args
        self.dist_mgr = dist_mgr
        self.dist = dist_mgr.info

        self.result_dir = self._ensure_result_dir(args.result_directory, args.dry_run)
        self.checkpointer = Checkpointer(self.result_dir, self.dist)

        self.conf = self._load_conf(args.conf_path)
        self._apply_conf_overrides()

        self.data_bundle = self._build_data()
        self.model_mgr = ModelManager(self.conf, self.dist, args.gpu_ids, self.dist.local_rank)

        self.optimizer = OptimFactory.adamw(self.model_mgr.model, args.learning_rate, args.weight_decay)
        if args.scheduler:
            args.scheduler.initialize(self.optimizer)
        self.scheduler = args.scheduler

        self.history = History.load(self.result_dir)
        self.epoch_history = EpochHistory(self.result_dir)

        self.epoch = 0
        self.epoch_resume_extra: Optional[dict] = None

        self._maybe_resume()
        self.dist_mgr.barrier()
        self._write_training_info()

    def _load_conf(self, path: str) -> Dict[str, Any]:
        """Load YAML config."""
        with open(path, "r") as f:
            conf = yaml.safe_load(f)
        if self.dist.is_main:
            print("Config:\n", conf, "\n")
        return conf

    def _apply_conf_overrides(self) -> None:
        """Set conf flags needed for training terms."""
        if "harmonic_net" in self.conf and self.args.train_term_def.names():
            self.conf["harmonic_net_return_terms"] = True

        if self.conf.get("external_embedding_channels") is None and (self.args.embedding_filename and self.args.embedding_filename != "embeddings.npy"):
            if self.dist.is_main:
                print("WARNING: external embeddings usually should use graph-network-ext network")

    def _build_data(self) -> DataBundle:
        """Construct datasets/loaders and add optional dataset features."""
        dm = DataModule(
            directory_path=self.args.directory_path,
            subsetpdbs=self.args.subsetpdbs,
            val_ratio=self.args.val_ratio,
            batch_size=self.args.batch_size,
            atoms_per_call=self.args.atoms_per_call,
            enable_shuffle=self.args.enable_shuffle,
            dataset_chunk_size=self.args.dataset_chunk_size,
            use_npfile=self.args.use_npfile,
            embedding_filename=self.args.embedding_filename,
            energy_matching=self.args.energy_matching,
            dist_info=self.dist,
        )
        bundle = dm.build()

        if "sequence_basis_radius" in self.conf:
            if self.dist.is_main:
                print(f"Adding sequences to dataset... (sequence_basis_radius={self.conf['sequence_basis_radius']})")
            for d in bundle.datasets:
                d.build_sequences()

        extra_terms = []
        if "harmonic_net" in self.conf:
            if self.dist.is_main:
                print(f"Adding classical terms to dataset... (harmonic_net={self.conf['harmonic_net']})")
            for d in bundle.datasets:
                d.build_classical_terms()

            if self.args.train_term_def.names():
                extra_terms = self.args.train_term_def.names()
                if self.dist.is_main:
                    print(f"Loading additional trained terms: {extra_terms}")
                    print(f"  Term Scales: {[self.args.train_term_def.scale(i) for i in extra_terms]}")
                    print(f"  Term Angle Wrap: {[self.args.train_term_def.wrap(i) for i in extra_terms]}")
                for d in bundle.datasets:
                    d.load_frame_terms(extra_terms)

        if self.args.use_force_weights:
            if self.dist.is_main:
                print("Loading forces weights...")
            for d in bundle.datasets:
                d.load_frame_terms(["forces_weights"])

        self.extra_train_terms = extra_terms
        if self.dist.is_main:
            print()
        return bundle

    def _ensure_result_dir(self, result_dir: Optional[str], dry_run: bool) -> str:
        """Create or validate result directory (main process only)."""
        if result_dir and os.path.exists(os.path.join(result_dir, "checkpoint.pth")):
            return result_dir
        if result_dir and os.path.exists(os.path.join(result_dir, "checkpoint-mini.pth")):
            return result_dir

        if result_dir is None:
            result_dir = "../data/result-" + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

        if os.path.exists(result_dir):
            info_path = os.path.join(result_dir, "training_info.json")
            if os.path.exists(info_path):
                if self.dist.is_main:
                    print("Re-initializing:", result_dir)
                return result_dir
            raise RuntimeError("Model directory exists but doesn't contain a checkpoint.pth or training_info.json file")

        if self.dist.is_main and not dry_run:
            os.makedirs(result_dir, exist_ok=False)
            print("Created:", result_dir)
        return result_dir

    def _maybe_resume(self) -> None:
        """Resume from checkpoint if present."""
        ckpt_path = self.checkpointer.find_resume_checkpoint()
        if self.dist.is_main:
            print("checkpoint_path", ckpt_path)

        if not ckpt_path:
            self.epoch = 0
            if self.dist.is_main:
                print("Saving to:", self.result_dir)
            return

        if self.dist.is_main:
            print("Resuming:", self.result_dir)

        device = self.model_mgr.device
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

        model_target = self.model_mgr.state_target_for_loading()
        model_util.load_state_dict_with_rename(model_target, ckpt["state_dict"])

        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        else:
            if self.dist.is_main:
                print("  No optimizer in checkpoint, resetting...")

        if self.scheduler and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        self.epoch_resume_extra = ckpt.get("extra")
        self.epoch = int(ckpt.get("epoch", 0))

        if self.epoch > 0:
            self.history = History.load(self.result_dir)
            self.epoch_history = EpochHistory(self.result_dir)

        if self.dist.is_main:
            print("Saving to:", self.result_dir)

    def _write_training_info(self) -> None:
        """Write training info and priors (main process only)."""
        if self.scheduler:
            lr_sched_repr = repr(self.scheduler)
        else:
            lr_sched_repr = None
            for g in self.optimizer.param_groups:
                g["lr"] = self.args.learning_rate

        self.checkpointer.write_training_info(
            epoch=self.epoch,
            directory_path=self.args.directory_path,
            pdb_list=self.data_bundle.pdb_list,
            params={
                "weight_decay": self.args.weight_decay,
                "learning_rate": self.args.learning_rate,
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                "energy_weight": self.args.energy_weight,
                "force_weight": self.args.force_weight,
                "embedding_filename": self.args.embedding_filename or "embeddings.npy",
                "world_size": self.dist.world_size if self.dist.world_size else 1,
                "lr_scheduler_repr": lr_sched_repr,
                "dry_run": self.args.dry_run,
            },
        )

    def run(self) -> None:
        """Main training loop."""
        if self.scheduler and self.scheduler.is_annealing():
            self.args.early_stopping = -1

        first_es_epoch = self.epoch if self.args.reset_early_stopping else 0
        verbose_loss_report = sys.stdout.isatty()

        criterion = nn.MSELoss()
        term_criterion = nn.MSELoss(reduction="none")

        while self.epoch < self.args.epochs:
            self._set_sampler_epoch(self.epoch)

            t0 = time.time()
            self.model_mgr.train()

            train_metrics = self._train_one_epoch(
                criterion=criterion,
                term_criterion=term_criterion,
                verbose=verbose_loss_report,
            )

            self._append_train_metrics(train_metrics)
            self.model_mgr.eval()

            val_metrics = self._validate_one_epoch(criterion=criterion, term_criterion=term_criterion)
            self.history.val.append(val_metrics["val_loss"])

            if self.scheduler:
                self.scheduler.step(val_metrics["val_loss"])

            if self.dist.is_main:
                self._log_epoch(train_metrics, val_metrics, t0)

            should_stop = False
            if self.dist.is_main:
                should_stop = check_early_stopping(self.history.val[first_es_epoch:], patience=self.args.early_stopping)
            should_stop = self.dist_mgr.broadcast_bool(should_stop, src=0, device=self.model_mgr.device)

            self._save_epoch(train_metrics, val_metrics)
            self.dist_mgr.barrier()

            if should_stop:
                if self.dist.is_main:
                    print("Early stopping triggered.")
                break

            self.epoch += 1

    def _set_sampler_epoch(self, epoch: int) -> None:
        """Set epoch on DistributedSamplers for determinism."""
        if not self.dist.enabled:
            return
        for s in self.data_bundle.train_samplers:
            if s is not None:
                s.set_epoch(epoch)

    def _train_one_epoch(self, criterion, term_criterion, verbose: bool) -> Dict[str, Any]:
        """Train for one epoch, returning aggregated metrics."""
        args = self.args
        model = self.model_mgr.parallel
        device_out = self.model_mgr.device_output

        train_loss = 0.0
        train_energy_loss = 0.0
        train_force_loss = 0.0
        num_cal = 0.0

        epoch_offset = 0
        mini_train_loss = 0.0
        mini_num_cal = 0.0

        train_term_losses = {k: 0.0 for k in self.extra_train_terms}
        train_term_num_cal = {k: 0 for k in self.extra_train_terms}

        if self.epoch_resume_extra:
            if self.dist.is_main:
                print("Resuming epoch...")
            train_loss = float(self.epoch_resume_extra["train_loss"])
            num_cal = float(self.epoch_resume_extra["num_cal"])
            epoch_offset = int(self.epoch_resume_extra["i"])
            self.epoch_resume_extra = None

        if self.dist.is_main:
            it = tqdm(
                enumerate(self.data_bundle.train),
                desc=f"Training ({self.epoch}/{args.epochs})",
                total=len(self.data_bundle.train),
                dynamic_ncols=True,
                miniters=1,
            )
        else:
            it = enumerate(self.data_bundle.train)

        for i, batch in it:
            if i < epoch_offset:
                continue
            if epoch_offset and i == epoch_offset and self.dist.is_main and hasattr(it, "write"):
                it.write(f"Resumed epoch at batch {i}")

            if args.mini_epoch_size and i > 0 and (i % args.mini_epoch_size) == 0:
                self._save_mini_checkpoint(i, train_loss, num_cal, mini_train_loss, mini_num_cal, len(self.data_bundle.train))
                mini_train_loss, mini_num_cal = 0.0, 0.0

            total_batch_size = sum([sb["force"].numel() for sb in batch])
            num_cal += total_batch_size
            mini_num_cal += total_batch_size

            total_term_batch_size = {k: sum([sb[k].numel() for sb in batch]) for k in train_term_losses}
            for k in train_term_num_cal:
                train_term_num_cal[k] += total_term_batch_size[k]

            self.optimizer.zero_grad()

            for sub_batch in batch:
                loss, energy_loss, force_loss, term_loss_dict = self._compute_loss_for_sub_batch(
                    sub_batch=sub_batch,
                    criterion=criterion,
                    term_criterion=term_criterion,
                    total_batch_size=total_batch_size,
                    total_term_batch_size=total_term_batch_size,
                    device_out=device_out,
                    model=model,
                )

                train_force_loss += float(force_loss) * total_batch_size
                if args.energy_matching:
                    train_energy_loss += float(energy_loss) * total_batch_size

                delta_loss = float(loss) * total_batch_size
                train_loss += delta_loss
                mini_train_loss += delta_loss

                for k, v in term_loss_dict.items():
                    train_term_losses[k] += float(v) * total_term_batch_size[k]

                loss.backward()

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step_batch(self.epoch + i / len(self.data_bundle.train))

            if args.dry_run:
                if self.dist.is_main:
                    print("\nDry run OK!")
                sys.exit(0)

            if verbose and self.dist.is_main and hasattr(it, "set_description"):
                desc = [f"Training ({self.epoch}/{args.epochs}) (T={train_loss/num_cal:.4f}"]
                for tname in train_term_losses:
                    desc.append(f"{tname}={train_term_losses[tname]/train_term_num_cal[tname]:.4f}")
                it.set_description(", ".join(desc) + ")")

        metrics = {
            "train_loss_sum": train_loss,
            "train_energy_loss_sum": train_energy_loss,
            "train_force_loss_sum": train_force_loss,
            "num_cal": num_cal,
            "train_term_losses_sum": train_term_losses,
            "train_term_num_cal": train_term_num_cal,
        }
        return self._aggregate_train_metrics(metrics)

    def _compute_loss_for_sub_batch(
        self,
        sub_batch: Dict[str, Any],
        criterion,
        term_criterion,
        total_batch_size: int,
        total_term_batch_size: Dict[str, int],
        device_out,
        model: nn.Module,
    ):
        """Compute loss (and side losses) for a single sub-batch."""
        args = self.args

        force = sub_batch.pop("force").reshape(-1, sub_batch["force"].shape[-1]).to(device_out)

        force_weights = None
        if args.use_force_weights:
            force_weights = sub_batch.pop("forces_weights").reshape(-1).to(device_out)

        energy = None
        if args.energy_matching:
            energy = sub_batch.pop("energy").reshape(-1, sub_batch["energy"].shape[-1]).to(device_out)

        term_targets = {}
        for k in self.extra_train_terms:
            term_targets[k] = sub_batch.pop(k).flatten().to(device_out)

        out_energy, out_force, extra = model(**sub_batch)

        sub_batch_size = force.numel()
        energy_loss = torch.tensor(0.0, device=out_force.device)
        if args.energy_matching:
            energy_loss = criterion(out_energy, energy) * (sub_batch_size / total_batch_size)

        force_loss = term_criterion(out_force, force)
        if force_weights is not None:
            force_loss = force_loss * force_weights[:, None]
        force_loss = force_loss.mean() * (sub_batch_size / total_batch_size)

        loss = args.energy_weight * energy_loss + args.force_weight * force_loss

        term_loss_dict: Dict[str, Tensor] = {}
        for k in self.extra_train_terms:
            if self.args.train_term_def.wrap(k):
                tl = (extra[k] - term_targets[k] + torch.pi) % (2 * torch.pi) - torch.pi
                tl = tl ** 2
            else:
                tl = term_criterion(extra[k], term_targets[k])

            tl = tl / total_term_batch_size[k]
            tl = tl * (term_targets[k] >= -10).float()
            tl = torch.sum(tl)

            loss = loss + tl * self.args.train_term_def.scale(k)
            term_loss_dict[k] = tl

        return loss, energy_loss, force_loss, term_loss_dict

    def _aggregate_train_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """All-reduce training metrics across processes."""
        if not self.dist.enabled:
            return metrics

        device = self.model_mgr.device
        metrics["train_loss_sum"] = self.dist_mgr.all_reduce_sum(metrics["train_loss_sum"], device)
        metrics["num_cal"] = self.dist_mgr.all_reduce_sum(metrics["num_cal"], device)
        metrics["train_energy_loss_sum"] = self.dist_mgr.all_reduce_sum(metrics["train_energy_loss_sum"], device)
        metrics["train_force_loss_sum"] = self.dist_mgr.all_reduce_sum(metrics["train_force_loss_sum"], device)

        for k in metrics["train_term_losses_sum"]:
            metrics["train_term_losses_sum"][k] = self.dist_mgr.all_reduce_sum(metrics["train_term_losses_sum"][k], device)
            metrics["train_term_num_cal"][k] = int(self.dist_mgr.all_reduce_sum(float(metrics["train_term_num_cal"][k]), device))

        return metrics

    def _append_train_metrics(self, train_metrics: Dict[str, Any]) -> None:
        """Append normalized train/energy/force losses to history."""
        num_cal = train_metrics["num_cal"]
        self.history.train.append(train_metrics["train_loss_sum"] / num_cal)
        self.history.energy.append(train_metrics["train_energy_loss_sum"] / num_cal)
        self.history.force.append(train_metrics["train_force_loss_sum"] / num_cal)

    def _validate_one_epoch(self, criterion, term_criterion) -> Dict[str, Any]:
        """Validate for one epoch, returning aggregated metrics."""
        args = self.args
        model = self.model_mgr.parallel
        device_out = self.model_mgr.device_output

        val_loss = 0.0
        num_cal = 0.0

        val_term_losses = {k: 0.0 for k in self.extra_train_terms}
        val_term_num_cal = {k: 0 for k in self.extra_train_terms}

        if self.dist.is_main:
            it = tqdm(self.data_bundle.val, desc=f"Validation ({self.epoch}/{args.epochs})", total=len(self.data_bundle.val), dynamic_ncols=True)
        else:
            it = self.data_bundle.val

        for batch in it:
            total_batch_size = sum([sb["force"].numel() for sb in batch])
            num_cal += total_batch_size

            total_term_batch_size = {k: sum([sb[k].numel() for sb in batch]) for k in val_term_losses}
            for k in val_term_num_cal:
                val_term_num_cal[k] += total_term_batch_size[k]

            for sub_batch in batch:
                force = sub_batch.pop("force").reshape(-1, sub_batch["force"].shape[-1]).to(device_out)

                force_weights = None
                if args.use_force_weights:
                    force_weights = sub_batch.pop("forces_weights").reshape(-1).to(device_out)

                energy = None
                if args.energy_matching:
                    energy = sub_batch.pop("energy").reshape(-1, sub_batch["energy"].shape[-1]).to(device_out)

                term_targets = {}
                for k in self.extra_train_terms:
                    term_targets[k] = sub_batch.pop(k).flatten().to(device_out)

                out_energy, out_force, extra = model(**sub_batch)

                sub_batch_size = force.numel()
                energy_loss = torch.tensor(0.0, device=out_force.device)
                if args.energy_matching:
                    energy_loss = criterion(out_energy, energy) * (sub_batch_size / total_batch_size)

                force_loss = term_criterion(out_force, force)
                if force_weights is not None:
                    force_loss = force_loss * force_weights[:, None]
                force_loss = force_loss.mean() * (sub_batch_size / total_batch_size)

                loss = args.energy_weight * energy_loss + args.force_weight * force_loss
                val_loss += float(loss) * total_batch_size

                for k in self.extra_train_terms:
                    if self.args.train_term_def.wrap(k):
                        tl = (extra[k] - term_targets[k] + torch.pi) % (2 * torch.pi) - torch.pi
                        tl = tl ** 2
                    else:
                        tl = term_criterion(extra[k], term_targets[k])

                    tl = tl / total_term_batch_size[k]
                    tl = tl * (term_targets[k] >= -10).float()
                    tl = torch.sum(tl)
                    val_term_losses[k] += float(tl) * total_term_batch_size[k]

        if self.dist.enabled:
            device = self.model_mgr.device
            val_loss = self.dist_mgr.all_reduce_sum(val_loss, device)
            num_cal = self.dist_mgr.all_reduce_sum(num_cal, device)
            for k in val_term_losses:
                val_term_losses[k] = self.dist_mgr.all_reduce_sum(val_term_losses[k], device)
                val_term_num_cal[k] = int(self.dist_mgr.all_reduce_sum(float(val_term_num_cal[k]), device))

        return {
            "val_loss_sum": val_loss,
            "num_cal": num_cal,
            "val_loss": val_loss / num_cal,
            "val_term_losses_sum": val_term_losses,
            "val_term_num_cal": val_term_num_cal,
        }

    def _save_mini_checkpoint(self, batch_i: int, train_loss: float, num_cal: float, mini_train_loss: float, mini_num_cal: float, epoch_len: int) -> None:
        """Save and rotate mini-checkpoint (main process only)."""
        tmp = os.path.join(self.result_dir, f"checkpoint-{self.epoch}-{batch_i}.pth")
        self.checkpointer.save(
            path=tmp,
            epoch=self.epoch,
            model_state=self.model_mgr.state_dict_for_saving(),
            optimizer=self.optimizer,
            model_conf=self.conf,
            scheduler=self.scheduler,
            extra={"train_loss": train_loss, "num_cal": num_cal, "i": batch_i},
        )

        if not self.dist.is_main:
            return

        os.replace(tmp, os.path.join(self.result_dir, "checkpoint-mini.pth"))

        self.epoch_history.update(
            f"{self.epoch}-{batch_i}",
            {
                "train_loss": train_loss / num_cal,
                "mini_train_loss": (mini_train_loss / mini_num_cal) if mini_num_cal else 0.0,
                "epoch_len": epoch_len,
                "lr": [g["lr"] for g in self.optimizer.param_groups],
            },
        )

    def _log_epoch(self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any], t0: float) -> None:
        """Print epoch summary and write epoch_history.json (main process only)."""
        entry = {
            "train_loss": self.history.train[-1],
            "val_loss": self.history.val[-1],
            "energy_loss": self.history.energy[-1],
            "force_loss": self.history.force[-1],
            "epoch_len": len(self.data_bundle.train),
            "lr": [g["lr"] for g in self.optimizer.param_groups],
        }

        for k in self.extra_train_terms:
            entry[f"train_loss_{k}"] = train_metrics["train_term_losses_sum"][k] / max(1, train_metrics["train_term_num_cal"][k])
            entry[f"val_loss_{k}"] = val_metrics["val_term_losses_sum"][k] / max(1, val_metrics["val_term_num_cal"][k])

        self.epoch_history.update(str(self.epoch), entry)

        print(
            f"Epoch {self.epoch} - Train Loss: {self.history.train[-1]} - Val Loss: {self.history.val[-1]} - time: {round(time.time() - t0, 2)}s"
        )
        if self.epoch > 0:
            print(f"  ∆Train: {self.history.train[-1]-self.history.train[-2]} - ∆Val: {self.history.val[-1]-self.history.val[-2]}")

        for k in self.extra_train_terms:
            print(f"  Train {k} loss={entry[f'train_loss_{k}']:.4f}")
            print(f"  Val {k} loss={entry[f'val_loss_{k}']:.4f}")

    def _save_epoch(self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]) -> None:
        """Save checkpoint/history (main process only), with barriers handled by caller."""
        tmp = os.path.join(self.result_dir, f"checkpoint-{self.epoch}.pth")

        self.checkpointer.save(
            path=tmp,
            epoch=self.epoch + 1,
            model_state=self.model_mgr.state_dict_for_saving(),
            optimizer=self.optimizer,
            model_conf=self.conf,
            scheduler=self.scheduler,
        )

        if not self.dist.is_main:
            return

        mini = os.path.join(self.result_dir, "checkpoint-mini.pth")
        if os.path.exists(mini):
            os.unlink(mini)

        main = os.path.join(self.result_dir, "checkpoint.pth")
        if self.args.checkpoint_save and (self.epoch % self.args.checkpoint_save == 0):
            shutil.copyfile(tmp, main)
        else:
            os.replace(tmp, main)

        best = os.path.join(self.result_dir, "checkpoint-best.pth")
        if self.history.val[-1] <= float(np.min(self.history.val)):
            shutil.copyfile(main, best)

        self.history.save(self.result_dir)
        print("  Checkpoint saved.")

# ----------------------------- CLI glue ------------------------------------

def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Train a CGSchNet network with distributed support (refactored)")
    p.add_argument("input", help="Processed data to train on ")
    p.add_argument("result", default=None, nargs="?", help="Checkpoint directory to continue")
    p.add_argument("-c", "--config", default="../configs/config.yaml", type=str)

    p.add_argument("--gpus", default=None, type=str, help='List of GPUs (e.g. "0,1,2") or "cpu"')
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--apc", "--atoms-per-call", dest="apc", type=int, default=None)

    p.add_argument("--cos-anneal", default=None, help='Cosine anneal: "T_0,T_mult"')
    p.add_argument("--cos-lr", default=None, help='Cosine LR: "T_max,eta_min"')
    p.add_argument("--exp-lr", default=None, help='Exponential LR: "gamma"')
    p.add_argument("--plateau-lr", default=None, help='Plateau LR: "factor,patience,threshold,min_lr"')

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reset-early-stopping", action="store_true")
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--mini-epoch", type=int, default=None)
    p.add_argument("--early-stopping", type=int, default=1)
    p.add_argument("--checkpoint-save", type=int, default=10)

    p.add_argument("--subsetpdbs", default="ok_list.txt", type=str)
    p.add_argument("--energy-weight", default=0.0, type=float)
    p.add_argument("--force-weight", default=1.0, type=float)

    p.add_argument("--term-def", default=None, type=str)
    p.add_argument("--embedding", type=str, default=None)
    p.add_argument("--chunk-dataset", type=int, default=None)
    p.add_argument("--npfile", action="store_true")
    p.add_argument("--use-force-weights", action="store_true")

    p.add_argument("--distributed", action="store_true")
    return p.parse_args()

def relax_open_file_limit() -> None:
    """Raise RLIMIT_NOFILE as much as possible."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

def parse_gpu_ids(gpus_arg: Optional[str]):
    """Parse --gpus argument into "cpu" or list[int]."""
    if not gpus_arg:
        return "cpu"
    if gpus_arg == "cpu":
        return "cpu"
    return [int(i) for i in gpus_arg.strip().split(",")]

def build_train_args(args) -> TrainArgs:
    """Translate argparse args into TrainArgs dataclass."""
    lr_scheduler = SchedulerFactory.from_args(args)
    train_term_def = TermDef(path=args.term_def) if args.term_def else TermDef()

    gpu_ids = parse_gpu_ids(args.gpus)
    energy_matching = args.energy_weight != 0.0

    return TrainArgs(
        directory_path=args.input,
        result_directory=args.result,
        conf_path=args.config,
        gpu_ids=gpu_ids,
        weight_decay=args.wd,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch,
        val_ratio=args.val_ratio,
        atoms_per_call=args.apc,
        scheduler=lr_scheduler,
        dry_run=args.dry_run,
        reset_early_stopping=args.reset_early_stopping,
        enable_shuffle=not args.no_shuffle,
        mini_epoch_size=args.mini_epoch,
        early_stopping=args.early_stopping,
        checkpoint_save=args.checkpoint_save,
        subsetpdbs=args.subsetpdbs,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        energy_matching=energy_matching,
        train_term_def=train_term_def,
        embedding_filename=args.embedding,
        dataset_chunk_size=args.chunk_dataset,
        use_npfile=args.npfile,
        use_force_weights=args.use_force_weights,
    )

def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA is not available, please run on a machine with CUDA or use --gpus cpu"
    assert os.path.isdir(args.input), f"Input directory does not exist: {args.input}"
    assert os.path.isfile(args.config), f"Config file does not exist: {args.config}"
    assert args.checkpoint_save >= 0

    relax_open_file_limit()

    dist_mgr = DistributedManager(enable=args.distributed)
    dist_mgr.setup()

    train_args = build_train_args(args)

    try:
        trainer = Trainer(train_args, dist_mgr)
        trainer.run()
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        sys.exit(1)
    finally:
        dist_mgr.cleanup()

if __name__ == "__main__":
    main()

