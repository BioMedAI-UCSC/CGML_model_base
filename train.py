#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
import numpy as np
from module.torchmdnet.model import create_model
from module import dataset
from module import model_util
from module.lr_scheduler_wrappers import *

import os
import json
import time
from tqdm import tqdm
import datetime
import shutil
import resource
import sys
import traceback
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from torch import Tensor

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# ===========================================================================
# Distributed context
# ===========================================================================

@dataclass
class DistributedContext:
    rank: Optional[int] = None
    world_size: Optional[int] = None
    local_rank: Optional[int] = None

    @property
    def is_main(self) -> bool:
        return self.rank is None or self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size is not None and self.world_size > 1

    def barrier(self):
        if self.is_distributed:
            dist.barrier()

    def all_reduce_scalar(self, value: float, device) -> float:
        if not self.is_distributed:
            return value
        t = torch.tensor(value, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item()

    def broadcast_bool(self, value: bool, device) -> bool:
        if not self.is_distributed:
            return value
        t = torch.tensor(1 if value else 0, device=device)
        dist.broadcast(t, src=0)
        return bool(t.item())


def setup_distributed() -> DistributedContext:
    """Auto-detect torchrun or SLURM environment and initialize NCCL process group."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    else:
        return DistributedContext()

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ===========================================================================
# Configuration dataclasses
# ===========================================================================

@dataclass
class LossConfig:
    energy_matching: bool
    energy_weight: float
    force_weight: float
    train_term_def: 'TermDef'
    extra_term_keys: List[str]
    use_force_weights: bool


@dataclass
class EpochMetrics:
    """Aggregated (summed) metrics for one epoch. Divide by num_cal to get means."""
    total_loss: float = 0.0
    energy_loss: float = 0.0
    force_loss: float = 0.0
    num_cal: int = 0
    term_losses: Dict[str, float] = field(default_factory=dict)
    term_num_cal: Dict[str, int] = field(default_factory=dict)

    def mean_loss(self) -> float:
        return self.total_loss / self.num_cal if self.num_cal > 0 else 0.0

    def mean_energy_loss(self) -> float:
        return self.energy_loss / self.num_cal if self.num_cal > 0 else 0.0

    def mean_force_loss(self) -> float:
        return self.force_loss / self.num_cal if self.num_cal > 0 else 0.0

    def mean_term_loss(self, k: str) -> float:
        n = self.term_num_cal.get(k, 0)
        return self.term_losses.get(k, 0.0) / n if n > 0 else 0.0


# ===========================================================================
# Pure utilities
# ===========================================================================

def flatten_first(t):
    """Flatten the first two dimensions of a tensor."""
    if t is None:
        return t
    if len(t.shape) < 2:
        return t
    return t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])


def make_term_offsets(lengths, term_lengths):
    result = []
    count = 0
    repeats = len(term_lengths) // len(lengths)
    lengths = np.tile(lengths, repeats)
    assert len(lengths) == len(term_lengths)
    for off, nterms in zip(lengths, term_lengths):
        result.append(torch.full((nterms, 1), count, dtype=torch.long))
        count += off
    return torch.cat(result)


def deterministic_shuffle(target, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n=len(target), generator=generator, device="cpu")
    return [target[i] for i in indices]


def check_early_stopping(val_list, patience=1) -> bool:
    """Return True if val loss increased patience+1 consecutive epochs. patience<0 disables."""
    if patience < 0:
        return False
    if len(val_list) < patience + 2:
        return False
    check_range = np.array(val_list[-(patience + 2):])
    if np.all((check_range[1:] - check_range[:-1]) > 0):
        print(f"Validation loss increased {patience + 1} times, stopping...")
        return True
    return False


def should_decay(param_name: str) -> bool:
    parts = param_name.split('.')
    assert len(parts) > 0
    if parts[-1] == "bias":
        return False
    if parts[-2] == "embedding":
        return False
    if parts[-2] == "distance_expansion":
        return False
    assert parts[-1] == "weight"
    return True


# ===========================================================================
# Model classes
# ===========================================================================

class BatchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pos, lengths, **kwargs) -> Tuple[Tensor, Tensor]:
        batch_nums = dataset.make_batch_nums(len(pos), lengths)
        batch_nums = batch_nums.to(pos.device)
        assert batch_nums.device == pos.device
        kwargs["pos"] = pos
        for k, v in kwargs.items():
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
            result = [*result, {}]
        return result  # pyright: ignore[reportReturnType]


class TermDef:
    def __init__(self, path=None, conf=None):
        self.scales: Dict[str, float] = {}
        self.angle_wrap: Dict[str, bool] = {}
        if path:
            with open(path, 'r') as f:
                conf = yaml.safe_load(f)
        if conf:
            for k, v in conf.items():
                self.scales[k] = float(v["scale"]) if v is not None and "scale" in v else 1.0
                self.angle_wrap[k] = bool(v["angle_wrap"]) if v is not None and "angle_wrap" in v else False

    def get_names(self) -> List[str]:
        return list(self.scales.keys())

    def get_scale(self, name: str) -> float:
        return self.scales[name]

    def get_angle_wrap(self, name: str) -> bool:
        return self.angle_wrap[name]


class RoundRobinDataWrapper:
    def __init__(self, *iterables):
        self.iterables = iterables

    def __len__(self):
        return sum(map(len, self.iterables))

    def __iter__(self):
        iterators = map(iter, self.iterables)
        for num_active in range(len(self.iterables), 0, -1):
            iterators = itertools.cycle(itertools.islice(iterators, num_active))
            yield from map(next, iterators)


# ===========================================================================
# Data loading
# ===========================================================================

def gen_dataloaders(directory_path, pdb_list, energy_filename, embedding_filename,
                    use_npfile, enable_shuffle, val_ratio, batch_size, atoms_per_call,
                    num_workers, ddp: DistributedContext):
    """Build train/val DataLoaders for one PDB chunk, with optional DistributedSampler."""
    print("Dataset:", " ".join(pdb_list))
    all_data = dataset.ProteinDataset(
        directory_path, pdb_list,
        energy_file=energy_filename, embeddings_file=embedding_filename, use_npfile=use_npfile,
    )
    assert 0.0 < val_ratio < 1.0
    val_size = int(val_ratio * len(all_data))
    train_size = len(all_data) - val_size

    if enable_shuffle:
        gen = torch.Generator().manual_seed(12341234)
        val_idx, train_idx = torch.utils.data.random_split(
            torch.arange(len(all_data)), [val_size, train_size], generator=gen
        )  # pyright: ignore[reportArgumentType]
    else:
        train_idx = range(train_size)
        val_idx = range(train_size, train_size + val_size)

    train_set = torch.utils.data.Subset(all_data, train_idx)  # pyright: ignore[reportArgumentType]
    val_set = torch.utils.data.Subset(all_data, val_idx)  # pyright: ignore[reportArgumentType]
    collate_fn = dataset.ProteinBatchCollate(atoms_per_call)

    train_sampler = (
        DistributedSampler(train_set, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False)
        if ddp.is_distributed else None
    )
    val_sampler = (
        DistributedSampler(val_set, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False)
        if ddp.is_distributed else None
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=True, pin_memory=True, collate_fn=collate_fn, sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=True, pin_memory=True, collate_fn=collate_fn, sampler=val_sampler,
    )
    return all_data, train_loader, val_loader, train_sampler


def _load_pdb_list(directory_path, subsetpdbs, dataset_chunk_size):
    """Read, deduplicate, shuffle, and optionally chunk the PDB ID list."""
    if subsetpdbs is None:
        subsetpdbs = os.path.join(directory_path, "result", "ok_list.txt")
    with open(subsetpdbs, 'r') as f:
        pdb_list = f.read().split('\n')
    pdb_list = sorted(set(p for p in pdb_list if p))
    pdb_list = deterministic_shuffle(pdb_list, seed=47563537)
    if dataset_chunk_size is not None:
        pdb_lists = [pdb_list[i:i + dataset_chunk_size] for i in range(0, len(pdb_list), dataset_chunk_size)]
    else:
        pdb_lists = [pdb_list]
    return pdb_lists, pdb_list


def _build_all_dataloaders(pdb_lists, directory_path, energy_filename, embedding_filename,
                            use_npfile, enable_shuffle, val_ratio, batch_size, atoms_per_call,
                            num_workers,
                            ddp: DistributedContext):
    """Build datasets and merged RoundRobin loaders for all PDB chunks."""
    datasets, train_loaders, val_loaders, train_samplers = [], [], [], []
    for pdb_chunk in pdb_lists:
        ds, train_loader, val_loader, sampler = gen_dataloaders(
            directory_path, pdb_chunk, energy_filename, embedding_filename,
            use_npfile, enable_shuffle, val_ratio, batch_size, atoms_per_call, num_workers, ddp,
        )
        datasets.append(ds)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        train_samplers.append(sampler)
    train_data = RoundRobinDataWrapper(*train_loaders)
    val_data = RoundRobinDataWrapper(*val_loaders)
    return datasets, train_data, val_data, train_samplers


# ===========================================================================
# Model setup
# ===========================================================================

def _load_model_conf(conf_path, ddp: DistributedContext) -> dict:
    """Load and print the model architecture YAML config."""
    if conf_path is None:
        conf_path = "../configs/config.yaml"
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
    if ddp.is_main:
        print("Config:\n", conf, "\n")
    return conf


def _augment_datasets(conf, datasets, train_term_def: TermDef, use_force_weights: bool,
                       embedding_filename: str, ddp: DistributedContext) -> List[str]:
    """Add optional features (sequences, classical terms, force weights) to every dataset."""
    if conf.get("external_embedding_channels") is None and embedding_filename != "embeddings.npy":
        if ddp.is_main:
            print("WARNING: external embeddings usually should use graph-network-ext network")

    extra_train_terms: List[str] = []

    if "sequence_basis_radius" in conf:
        if ddp.is_main:
            print(f"Adding sequences to dataset... (sequence_basis_radius={conf['sequence_basis_radius']})")
        for d in datasets:
            d.build_sequences()

    if "harmonic_net" in conf:
        if ddp.is_main:
            print(f"Adding classical terms to dataset... (harmonic_net={conf['harmonic_net']})")
        for d in datasets:
            d.build_classical_terms()
        if train_term_def.get_names():
            harmonic_trained_terms = train_term_def.get_names()
            if ddp.is_main:
                print(f"Loading additional trained terms: {harmonic_trained_terms}")
                print(f"  Term Scales:     {[train_term_def.get_scale(t) for t in harmonic_trained_terms]}")
                print(f"  Term Angle Wrap: {[train_term_def.get_angle_wrap(t) for t in harmonic_trained_terms]}")
            for d in datasets:
                d.load_frame_terms(harmonic_trained_terms)
            extra_train_terms.extend(harmonic_trained_terms)

    if use_force_weights:
        if ddp.is_main:
            print("Loading force weights...")
        for d in datasets:
            d.load_frame_terms(["forces_weights"])

    return extra_train_terms


def build_parallel_model(model, gpu_ids, ddp: DistributedContext):
    """
    Wrap model for distributed/parallel execution.
    Returns (parallel_model, device, device_output).
      device        — where weights live; use for checkpoints and all_reduce
      device_output — where model outputs land; use for moving targets
    """
    wrapped = BatchWrapper(model)
    if ddp.is_distributed:
        device = torch.device(f'cuda:{ddp.local_rank}')
        model.to(device)
        parallel = DDP(wrapped, device_ids=[ddp.local_rank], output_device=ddp.local_rank,
                       find_unused_parameters=True)
        device_output = device
        if ddp.is_main:
            print(f"DDP: {ddp.world_size} GPUs (local_rank={ddp.local_rank})")
    elif gpu_ids == "cpu":
        device = torch.device("cpu")
        model.to(device)
        parallel = wrapped
        device_output = device
        print("Training on CPU")
    else:
        device = torch.device(f'cuda:{gpu_ids[0]}')
        model.to(device)
        parallel = nn.DataParallel(wrapped, device_ids=gpu_ids)
        device_output = parallel.output_device
        print(f"DataParallel: {len(parallel.device_ids)} GPU(s)")
    return parallel, device, device_output


# ===========================================================================
# Optimizer
# ===========================================================================

def _build_optimizer(model, weight_decay: float, learning_rate: float):
    """AdamW with weight decay applied only to non-embedding, non-bias parameters."""
    do_decay, dont_decay = [], []
    for name, param in model.named_parameters():
        (do_decay if should_decay(name) else dont_decay).append(param)
    return optim.AdamW(
        [{"params": do_decay, "weight_decay": weight_decay}, {"params": dont_decay}],
        lr=learning_rate,
    )


# ===========================================================================
# Checkpoint I/O
# ===========================================================================

def save_checkpoint(checkpoint_path, epoch, model, optimizer, model_conf, scheduler,
                    extra=None, ddp: Optional[DistributedContext] = None):
    """Save checkpoint. In distributed mode only rank 0 writes."""
    if ddp is not None and not ddp.is_main:
        return
    d = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "state_dict": model.state_dict(),
        "hyper_parameters": model_conf,
    }
    if scheduler:
        d["scheduler"] = scheduler.state_dict()
    if extra:
        d["extra"] = extra
    torch.save(d, checkpoint_path)


def _load_checkpoint_state(result_directory, model, optimizer, scheduler, device,
                            ddp: DistributedContext):
    """Find and load the best available checkpoint. Returns (epoch, epoch_resume)."""
    checkpoint_path = None
    if result_directory and os.path.exists(f'{result_directory}/checkpoint-mini.pth'):
        checkpoint_path = f'{result_directory}/checkpoint-mini.pth'
    elif result_directory and os.path.exists(f'{result_directory}/checkpoint.pth'):
        checkpoint_path = f'{result_directory}/checkpoint.pth'

    if ddp.is_main:
        print("checkpoint_path", checkpoint_path)

    if not checkpoint_path:
        return 0, None

    if ddp.is_main:
        print("Resuming:", result_directory)

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_util.load_state_dict_with_rename(model, checkpoint["state_dict"])

    if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    elif ddp.is_main:
        print("  No optimizer in checkpoint, resetting...")

    if scheduler and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint.get("epoch", 0), checkpoint.get("extra")


def _commit_epoch_checkpoint(epoch, model, optimizer, conf, scheduler, result_directory,
                              val_loss_list, history, checkpoint_save, ddp: DistributedContext):
    """Atomically promote the epoch checkpoint and optionally save a best/backup copy."""
    tmp_path = f'{result_directory}/checkpoint-{epoch}.pth'
    save_checkpoint(tmp_path, epoch + 1, model, optimizer, conf, scheduler, ddp=ddp)

    if not ddp.is_main:
        return

    if os.path.exists(f'{result_directory}/checkpoint-mini.pth'):
        os.unlink(f'{result_directory}/checkpoint-mini.pth')

    if checkpoint_save and (epoch % checkpoint_save == 0):
        shutil.copyfile(tmp_path, f'{result_directory}/checkpoint.pth')
    else:
        os.replace(tmp_path, f'{result_directory}/checkpoint.pth')

    if val_loss_list[-1] <= np.min(val_loss_list):
        shutil.copyfile(f'{result_directory}/checkpoint.pth', f'{result_directory}/checkpoint-best.pth')

    np.save(f'{result_directory}/history.npy', history)  # pyright: ignore[reportArgumentType]
    print("  Checkpoint saved.")


def _save_mini_epoch_checkpoint(epoch, i, model, optimizer, conf, scheduler,
                                 total_loss, num_cal, mini_train_loss, mini_num_cal,
                                 train_data, result_directory, epoch_history,
                                 ddp: DistributedContext):
    """Save a mid-epoch checkpoint and record the mini-epoch in epoch_history."""
    tmp_path = f'{result_directory}/checkpoint-{epoch}-{i}.pth'
    save_checkpoint(tmp_path, epoch, model, optimizer, conf, scheduler,
                    extra={"train_loss": total_loss, "num_cal": num_cal, "i": i}, ddp=ddp)
    if not ddp.is_main:
        return
    os.replace(tmp_path, f'{result_directory}/checkpoint-mini.pth')
    epoch_history[f"{epoch}-{i}"] = {
        "train_loss": total_loss / num_cal,
        "mini_train_loss": mini_train_loss / mini_num_cal,
        "epoch_len": len(train_data),
        "lr": [g['lr'] for g in optimizer.param_groups],
    }
    with open(os.path.join(result_directory, "epoch_history.json"), "w") as f:
        json.dump(epoch_history, f, indent=2)


# ===========================================================================
# Result directory and history
# ===========================================================================

def _setup_result_directory(result_directory, epoch, dry_run, ddp: DistributedContext) -> str:
    """Create the result directory for a fresh run, or validate it for resumption."""
    if epoch > 0:
        pass  # Resuming — directory already exists
    elif not os.path.exists(result_directory):
        if not dry_run and ddp.is_main:
            os.makedirs(result_directory, exist_ok=False)
        if ddp.is_main:
            print("Created:", result_directory)
    elif os.path.exists(f'{result_directory}/training_info.json'):
        if ddp.is_main:
            print("Re-initializing:", result_directory)
    else:
        raise RuntimeError("Model directory exists but doesn't contain a checkpoint.pth or training_info.json file")

    if ddp.is_main:
        print("Saving to:", result_directory)
    return result_directory


def _load_history(result_directory, epoch):
    """Load loss history arrays and epoch_history JSON from a previous run."""
    train_loss_list, val_loss_list, energy_loss_list, force_loss_list = [], [], [], []
    epoch_history: dict = {}

    if epoch > 0:
        history = np.load(f'{result_directory}/history.npy', allow_pickle=True).item()
        train_loss_list = history['train']
        val_loss_list = history['val']
        energy_loss_list = history['energy']
        force_loss_list = history['force']

    epoch_history_path = os.path.join(result_directory, "epoch_history.json")
    if os.path.exists(epoch_history_path):
        with open(epoch_history_path, "r") as f:
            epoch_history = json.load(f)

    return train_loss_list, val_loss_list, energy_loss_list, force_loss_list, epoch_history


def _write_training_info(result_directory, epoch, weight_decay, learning_rate, epochs, batch_size,
                          directory_path, pdb_list, energy_weight, force_weight, embedding_filename,
                          ddp: DistributedContext, scheduler, optimizer, dry_run):
    """Write training_info.json and copy prior files. Only runs on rank 0."""
    if not ddp.is_main:
        return

    training_info_path = os.path.join(result_directory, "training_info.json")
    training_info_dict: dict = {}

    if os.path.exists(training_info_path):
        with open(training_info_path, "r") as f:
            training_info_dict = json.load(f)
        if "input_directory" in training_info_dict:
            training_info_dict = {"0": training_info_dict}
    else:
        print("Path", training_info_path, "does not exist")

    training_info_dict[str(epoch)] = {
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_directory": directory_path,
        "pdbs": pdb_list,
        "energy_weight": energy_weight,
        "force_weight": force_weight,
        "embedding_filename": embedding_filename,
        "world_size": ddp.world_size if ddp.world_size else 1,
    }

    if scheduler:
        training_info_dict[str(epoch)]["lr_scheduler"] = repr(scheduler)
    else:
        for g in optimizer.param_groups:
            g['lr'] = learning_rate

    if not dry_run:
        with open(training_info_path, "w") as f:
            json.dump(training_info_dict, f, indent=2)
        prior_path = os.path.join(directory_path, "priors.yaml")
        if os.path.exists(prior_path):
            shutil.copy(prior_path, result_directory)
            shutil.copy(os.path.join(directory_path, "prior_params.json"), result_directory)


# ===========================================================================
# Per-epoch helpers
# ===========================================================================

def _set_sampler_epochs(train_samplers, epoch):
    """Notify DistributedSamplers of the current epoch so shuffling stays deterministic."""
    for sampler in train_samplers:
        if sampler is not None:
            sampler.set_epoch(epoch)


def _update_loss_history(train_loss_list, val_loss_list, energy_loss_list, force_loss_list,
                          train_metrics: EpochMetrics, val_metrics: EpochMetrics):
    """Append per-epoch means to the running history lists."""
    train_loss_list.append(train_metrics.mean_loss())
    energy_loss_list.append(train_metrics.mean_energy_loss())
    force_loss_list.append(train_metrics.mean_force_loss())
    val_loss_list.append(val_metrics.mean_loss())


def _log_epoch_results(ddp: DistributedContext, epoch, train_loss_list, val_loss_list,
                        train_metrics: EpochMetrics, val_metrics: EpochMetrics,
                        extra_train_terms, epoch_history, optimizer, t0, train_data,
                        result_directory):
    """Record epoch stats in epoch_history, write JSON, and print a summary."""
    if not ddp.is_main:
        return

    epoch_history[f"{epoch}"] = {
        "train_loss": train_loss_list[-1],
        "val_loss": val_loss_list[-1],
        "energy_loss": train_metrics.mean_energy_loss(),
        "force_loss": train_metrics.mean_force_loss(),
        "epoch_len": len(train_data),
        "lr": [g['lr'] for g in optimizer.param_groups],
    }
    for k in extra_train_terms:
        epoch_history[f"{epoch}"][f"train_loss_{k}"] = train_metrics.mean_term_loss(k)
        epoch_history[f"{epoch}"][f"val_loss_{k}"] = val_metrics.mean_term_loss(k)

    with open(os.path.join(result_directory, "epoch_history.json"), "w") as f:
        json.dump(epoch_history, f, indent=2)

    print(f"Epoch {epoch} - Train Loss: {train_loss_list[-1]} - Val Loss: {val_loss_list[-1]} - time: {round(time.time() - t0, 2)}s")
    if epoch > 0:
        print(f"  ∆Train: {train_loss_list[-1] - train_loss_list[-2]} - ∆Val: {val_loss_list[-1] - val_loss_list[-2]}")
    for k in extra_train_terms:
        print(f"  Train {k} loss={train_metrics.mean_term_loss(k):.4f}")
        print(f"  Val   {k} loss={val_metrics.mean_term_loss(k):.4f}")


def _check_early_stop(ddp: DistributedContext, val_loss_list, first_early_stopping_epoch,
                       early_stopping, device) -> bool:
    """Evaluate early stopping on rank 0 then broadcast the decision to all ranks."""
    should_stop = False
    if ddp.is_main:
        should_stop = check_early_stopping(val_loss_list[first_early_stopping_epoch:], patience=early_stopping)
    should_stop = ddp.broadcast_bool(should_stop, device)
    if should_stop and ddp.is_main:
        print("Early stopping triggered.")
    return should_stop


# ===========================================================================
# Forward pass (shared between train and val)
# ===========================================================================

def _process_sub_batch(sub_batch, parallel_model, criterion, term_criterion,
                        device_output, loss_cfg: LossConfig,
                        total_batch_size: int, total_term_batch_size: Dict[str, int]):
    """
    Forward one sub-batch through the model and compute weighted losses.
    Returns (loss_tensor, loss_sum, energy_loss_sum, force_loss_sum, term_loss_sums).
    *_sum values are pre-multiplied by total_batch_size for direct accumulation.
    Call loss_tensor.backward() in the training loop.
    """
    force = sub_batch.pop("force")
    force = force.reshape(-1, force.shape[-1]).to(device_output)

    force_weights = None
    if loss_cfg.use_force_weights:
        force_weights = sub_batch.pop("forces_weights").reshape(-1).to(device_output)

    energy = None
    if loss_cfg.energy_matching:
        energy = sub_batch.pop("energy")
        energy = energy.reshape(-1, energy.shape[-1]).to(device_output)

    term_targets = {k: sub_batch.pop(k).flatten().to(device_output) for k in loss_cfg.extra_term_keys}

    out_energy, out_force, extra = parallel_model(**sub_batch)

    sub_batch_size = force.numel()
    weight = sub_batch_size / total_batch_size

    energy_loss = torch.tensor(0.0)
    if loss_cfg.energy_matching:
        energy_loss = criterion(out_energy, energy) * weight

    fl = term_criterion(out_force, force)
    if force_weights is not None:
        fl = fl * force_weights[:, None]
    force_loss = fl.mean() * weight

    loss = loss_cfg.energy_weight * energy_loss + loss_cfg.force_weight * force_loss

    term_loss_sums: Dict[str, float] = {}
    for k in loss_cfg.extra_term_keys:
        td = loss_cfg.train_term_def
        if td.get_angle_wrap(k):
            tl = (extra[k] - term_targets[k] + torch.pi) % (2 * torch.pi) - torch.pi
            tl = tl ** 2
        else:
            tl = term_criterion(extra[k], term_targets[k])
        tl = tl / total_term_batch_size[k]
        tl = tl * (term_targets[k] >= -10).float()
        tl = torch.sum(tl)
        loss = loss + tl * td.get_scale(k)
        term_loss_sums[k] = tl.item() * total_term_batch_size[k]

    return loss, loss.item() * total_batch_size, energy_loss.item() * total_batch_size, force_loss.item() * total_batch_size, term_loss_sums


def _reduce_to_epoch_metrics(total_loss, energy_loss, force_loss, num_cal,
                              term_losses, term_num_cal, extra_term_keys,
                              ddp: DistributedContext, device) -> EpochMetrics:
    """All-reduce raw accumulators across ranks and package into EpochMetrics."""
    return EpochMetrics(
        total_loss=ddp.all_reduce_scalar(total_loss, device),
        energy_loss=ddp.all_reduce_scalar(energy_loss, device),
        force_loss=ddp.all_reduce_scalar(force_loss, device),
        num_cal=int(ddp.all_reduce_scalar(num_cal, device)),
        term_losses={k: ddp.all_reduce_scalar(term_losses[k], device) for k in extra_term_keys},
        term_num_cal={k: int(ddp.all_reduce_scalar(term_num_cal[k], device)) for k in extra_term_keys},
    )


# ===========================================================================
# Epoch runners
# ===========================================================================

def run_train_epoch(
    epoch, epochs, train_data, parallel_model, model,
    optimizer, scheduler, criterion, term_criterion,
    device, device_output, loss_cfg: LossConfig, ddp: DistributedContext,
    dry_run: bool, verbose_loss_report: bool,
    mini_epoch_size, epoch_resume, result_directory, conf, epoch_history,
) -> EpochMetrics:
    parallel_model.train()

    total_loss = 0.0
    energy_loss = 0.0
    force_loss = 0.0
    num_cal = 0
    mini_train_loss = 0.0
    mini_num_cal = 0
    epoch_offset = 0
    term_losses: Dict[str, float] = {k: 0.0 for k in loss_cfg.extra_term_keys}
    term_num_cal: Dict[str, int] = {k: 0 for k in loss_cfg.extra_term_keys}

    if epoch_resume:
        if ddp.is_main:
            print("Resuming epoch...")
        total_loss = float(epoch_resume["train_loss"])
        num_cal = int(epoch_resume["num_cal"])
        epoch_offset = int(epoch_resume["i"])

    # Setting miniters is required to keep the bar from stalling when skipping ahead on resume
    tqdm_iter = (
        tqdm(enumerate(train_data), desc=f"Training ({epoch}/{epochs})",
             total=len(train_data), dynamic_ncols=True, miniters=1)
        if ddp.is_main else enumerate(train_data)
    )

    for i, batch in tqdm_iter:
        if i < epoch_offset:
            continue
        elif epoch_offset and i == epoch_offset:
            if ddp.is_main and hasattr(tqdm_iter, 'write'):
                tqdm_iter.write(f"Resumed epoch at batch {i}")
        elif mini_epoch_size and i > 0 and i % mini_epoch_size == 0:
            _save_mini_epoch_checkpoint(
                epoch, i, model, optimizer, conf, scheduler,
                total_loss, num_cal, mini_train_loss, mini_num_cal,
                train_data, result_directory, epoch_history, ddp,
            )
            if ddp.is_main and hasattr(tqdm_iter, 'write'):
                tqdm_iter.write(f"Mini-epoch {epoch}-{i}: Train Loss: {total_loss / num_cal}")
            mini_train_loss = 0.0
            mini_num_cal = 0

        batch_size_total = sum(sb["force"].numel() for sb in batch)
        batch_term_sizes = {k: sum(sb[k].numel() for sb in batch) for k in loss_cfg.extra_term_keys}
        num_cal += batch_size_total
        mini_num_cal += batch_size_total
        for k in term_num_cal:
            term_num_cal[k] += batch_term_sizes[k]

        optimizer.zero_grad()
        for sub_batch in batch:
            loss, loss_sum, e_sum, f_sum, term_sums = _process_sub_batch(
                sub_batch, parallel_model, criterion, term_criterion,
                device_output, loss_cfg, batch_size_total, batch_term_sizes,
            )
            total_loss += loss_sum
            mini_train_loss += loss_sum
            energy_loss += e_sum
            force_loss += f_sum
            for k, v in term_sums.items():
                term_losses[k] += v
            loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step_batch(epoch + i / len(train_data))

        if dry_run:
            if ddp.is_main:
                print("\nDry run OK!")
            sys.exit(0)

        if verbose_loss_report and ddp.is_main and hasattr(tqdm_iter, 'set_description'):
            desc = [f"Training ({epoch}/{epochs}) (T={total_loss / num_cal:.4f}"]
            for t in loss_cfg.extra_term_keys:
                if term_num_cal[t] > 0:
                    desc.append(f"{t}={term_losses[t] / term_num_cal[t]:.4f}")
            tqdm_iter.set_description(", ".join(desc) + ")")

    return _reduce_to_epoch_metrics(
        total_loss, energy_loss, force_loss, num_cal, term_losses, term_num_cal,
        loss_cfg.extra_term_keys, ddp, device,
    )


def run_val_epoch(
    epoch, epochs, val_data, parallel_model,
    criterion, term_criterion, device, device_output,
    loss_cfg: LossConfig, ddp: DistributedContext,
) -> EpochMetrics:
    parallel_model.eval()

    total_loss = 0.0
    num_cal = 0
    term_losses: Dict[str, float] = {k: 0.0 for k in loss_cfg.extra_term_keys}
    term_num_cal: Dict[str, int] = {k: 0 for k in loss_cfg.extra_term_keys}

    val_iter = (
        tqdm(val_data, desc=f"Validation ({epoch}/{epochs})", total=len(val_data), dynamic_ncols=True)
        if ddp.is_main else val_data
    )

    with torch.no_grad():
        for batch in val_iter:
            batch_size_total = sum(sb["force"].numel() for sb in batch)
            batch_term_sizes = {k: sum(sb[k].numel() for sb in batch) for k in loss_cfg.extra_term_keys}
            num_cal += batch_size_total
            for k in term_num_cal:
                term_num_cal[k] += batch_term_sizes[k]

            for sub_batch in batch:
                _, loss_sum, _, _, term_sums = _process_sub_batch(
                    sub_batch, parallel_model, criterion, term_criterion,
                    device_output, loss_cfg, batch_size_total, batch_term_sizes,
                )
                total_loss += loss_sum
                for k, v in term_sums.items():
                    term_losses[k] += v

    return _reduce_to_epoch_metrics(
        total_loss, 0.0, 0.0, num_cal, term_losses, term_num_cal,
        loss_cfg.extra_term_keys, ddp, device,
    )


# ===========================================================================
# Training orchestration
# ===========================================================================

def train_model(
    directory_path, conf_path, result_directory, dry_run, gpu_ids,
    weight_decay, learning_rate, epochs, batch_size, val_ratio, atoms_per_call,
    scheduler, reset_early_stopping, enable_shuffle, mini_epoch_size, early_stopping,
    checkpoint_save, subsetpdbs, energy_weight, force_weight, energy_matching,
    train_term_def, embedding_filename, dataset_chunk_size, use_npfile, use_force_weights,
    num_workers,
    ddp: Optional[DistributedContext] = None,
):
    if ddp is None:
        ddp = DistributedContext()

    energy_filename = "tica_delta_energies.npy" if energy_matching else None
    if embedding_filename is None:
        embedding_filename = "embeddings.npy"

    pdb_lists, pdb_list = _load_pdb_list(directory_path, subsetpdbs, dataset_chunk_size)
    datasets, train_data, val_data, train_samplers = _build_all_dataloaders(
        pdb_lists, directory_path, energy_filename, embedding_filename,
        use_npfile, enable_shuffle, val_ratio, batch_size, atoms_per_call, num_workers, ddp,
    )

    conf = _load_model_conf(conf_path, ddp)
    if "harmonic_net" in conf and train_term_def.get_names():
        conf["harmonic_net_return_terms"] = True

    model = create_model(args=conf)
    parallel_model, device, device_output = build_parallel_model(model, gpu_ids, ddp)
    if ddp.is_main:
        print("Model:\n", model, "\n")

    extra_train_terms = _augment_datasets(conf, datasets, train_term_def, use_force_weights, embedding_filename, ddp)
    loss_cfg = LossConfig(
        energy_matching=energy_matching, energy_weight=energy_weight, force_weight=force_weight,
        train_term_def=train_term_def, extra_term_keys=extra_train_terms, use_force_weights=use_force_weights,
    )
    criterion = nn.MSELoss()
    term_criterion = nn.MSELoss(reduction="none")

    optimizer = _build_optimizer(model, weight_decay, learning_rate)
    if scheduler:
        scheduler.initialize(optimizer)

    epoch, epoch_resume = _load_checkpoint_state(result_directory, model, optimizer, scheduler, device, ddp)
    result_directory = _setup_result_directory(result_directory, epoch, dry_run, ddp)
    ddp.barrier()

    train_loss_list, val_loss_list, energy_loss_list, force_loss_list, epoch_history = _load_history(result_directory, epoch)
    _write_training_info(
        result_directory, epoch, weight_decay, learning_rate, epochs, batch_size,
        directory_path, pdb_list, energy_weight, force_weight, embedding_filename,
        ddp, scheduler, optimizer, dry_run,
    )

    if scheduler and scheduler.is_annealing():
        early_stopping = -1
    first_early_stopping_epoch = epoch if reset_early_stopping else 0
    verbose_loss_report = sys.stdout.isatty()

    while epoch < epochs:
        _set_sampler_epochs(train_samplers, epoch)
        t0 = time.time()

        train_metrics = run_train_epoch(
            epoch, epochs, train_data, parallel_model, model,
            optimizer, scheduler, criterion, term_criterion,
            device, device_output, loss_cfg, ddp,
            dry_run, verbose_loss_report,
            mini_epoch_size, epoch_resume, result_directory, conf, epoch_history,
        )
        epoch_resume = None

        val_metrics = run_val_epoch(
            epoch, epochs, val_data, parallel_model,
            criterion, term_criterion, device, device_output, loss_cfg, ddp,
        )

        _update_loss_history(train_loss_list, val_loss_list, energy_loss_list, force_loss_list,
                              train_metrics, val_metrics)

        if scheduler:
            scheduler.step(val_loss_list[-1])

        _log_epoch_results(ddp, epoch, train_loss_list, val_loss_list, train_metrics, val_metrics,
                            extra_train_terms, epoch_history, optimizer, t0, train_data, result_directory)

        if _check_early_stop(ddp, val_loss_list, first_early_stopping_epoch, early_stopping, device):
            break

        history = {"train": train_loss_list, "val": val_loss_list, "energy": energy_loss_list, "force": force_loss_list}
        _commit_epoch_checkpoint(epoch, model, optimizer, conf, scheduler, result_directory,
                                  val_loss_list, history, checkpoint_save, ddp)
        ddp.barrier()
        epoch += 1


# ===========================================================================
# Entry-point helpers: collection → validation → processing
# ===========================================================================

def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Train a CGSchNet network")
    parser.add_argument("input", help="Processed data to train on")
    parser.add_argument("result", help="Checkpoint directory to continue")
    parser.add_argument("-c", "--config", default="../configs/config.yaml", type=str, help="Path to model architecture config YAML")
    parser.add_argument("--gpus", default=None, type=str, help="List of GPUs to train on (e.g. \"0,1,2\") or \"cpu\"")
    parser.add_argument("--batch", type=int, default=50, help="The batch size to use")
    parser.add_argument("--epochs", type=int, default=25, help="The total number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio, should be between 0.0 and 1.0")
    parser.add_argument("--apc", "--atoms-per-call", type=int, default=None, help="Number of atoms to include in each sub-batch")
    parser.add_argument("--cos-anneal", default=None, help="Train using cosine annealing, parameters are \"T_0,T_mult\"")
    parser.add_argument("--cos-lr", default=None, help="Train using a cosine learning rate, parameters are \"T_max,eta_min\"")
    parser.add_argument("--exp-lr", default=None, help="Train using an exponential learning rate, parameter is \"gamma\"")
    parser.add_argument("--plateau-lr", default=None, help="Train using a plateau learning rate, parameters are \"factor,patience,threshold,min_lr\"")
    parser.add_argument("--dry-run", action="store_true", help="Do a dry run of the training loop but produce no output")
    parser.add_argument("--reset-early-stopping", action="store_true", help="Reset the early stopping check to start from the current epoch")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the training dataset")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers to use for data loading")
    parser.add_argument("--mini-epoch", type=int, default=None, help="Save a mini epoch after every n batches")
    parser.add_argument("--early-stopping", type=int, default=1, help="Epochs validation loss can increase before early stopping, or -1 to disable (default=1)")
    parser.add_argument("--checkpoint-save", type=int, default=10, help="Save a backup checkpoint every n epochs, 0 to disable (default=10)")
    parser.add_argument("--subsetpdbs", default=None, type=str, help="Change the pdbid list used when reading in the dataset (default=input_dir/result/ok_list.txt)")
    parser.add_argument("--energy-weight", default=0.0, type=float, help="Energy weighting for loss function")
    parser.add_argument("--force-weight", default=1.0, type=float, help="Force weighting for loss function")
    parser.add_argument("--term-def", default=None, type=str, help="Path to a term definition yaml file for additional loss terms during training")
    parser.add_argument("--embedding", type=str, default=None, help="Alternate file to load embeddings from (default: embeddings.npy)")
    parser.add_argument("--chunk-dataset", type=int, default=None, help="Break the dataset into chunks of n proteins per batch")
    parser.add_argument("--npfile", action="store_true", help="Use file loader instead of mmap to load dataset")
    parser.add_argument("--use-force-weights", action="store_true", default=False, help="Use per-bead force weights in training")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training (also auto-detected from RANK/SLURM_PROCID env vars)")
    return parser


def validate_config(cfg) -> None:
    assert torch.cuda.is_available(), "CUDA is not available, please run on a machine with CUDA or use --gpus cpu"
    assert os.path.isdir(cfg.input), f"Input directory does not exist: {cfg.input}"
    assert os.path.isdir(cfg.result), f"Result directory does not exist: {cfg.result}"
    assert os.path.isfile(cfg.config), f"Config file does not exist: {cfg.config}"
    assert cfg.checkpoint_save >= 0, "--checkpoint-save must be >= 0"
    n_schedulers = sum(s is not None for s in [cfg.cos_anneal, cfg.cos_lr, cfg.exp_lr, cfg.plateau_lr])
    assert n_schedulers <= 1, "At most one LR scheduler may be specified (--cos-anneal, --cos-lr, --exp-lr, --plateau-lr)"


def process_config(cfg) -> dict:
    if cfg.gpus == "cpu":
        gpu_ids = "cpu"
    elif cfg.gpus:
        gpu_ids = [int(i) for i in cfg.gpus.strip().split(",")]
    else:
        gpu_ids = "cpu"

    lr_scheduler = None
    if cfg.cos_anneal:
        T_0, T_mult = [int(i) for i in cfg.cos_anneal.split(",")]
        lr_scheduler = SchedulerWrapper_CosineAnnealingWarmRestarts(T_0, T_mult)
    elif cfg.cos_lr:
        T_max, eta_min = cfg.cos_lr.split(",")
        lr_scheduler = SchedulerWrapper_CosineAnnealingLR(int(T_max), float(eta_min))
    elif cfg.exp_lr:
        lr_scheduler = SchedulerWrapper_ExponentialLR(float(cfg.exp_lr))
    elif cfg.plateau_lr:
        factor, patience, threshold, min_lr = cfg.plateau_lr.split(",")
        lr_scheduler = SchedulerWrapper_ReduceLROnPlateau(float(factor), int(patience), float(threshold), float(min_lr))

    train_term_def = TermDef(path=cfg.term_def) if cfg.term_def is not None else TermDef()

    # Raise the OS file-descriptor limit — large datasets open ~4 files per molecule
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    return dict(
        directory_path=cfg.input,
        result_directory=cfg.result,
        conf_path=cfg.config,
        dry_run=cfg.dry_run,
        weight_decay=cfg.wd,
        learning_rate=cfg.lr,
        gpu_ids=gpu_ids,
        epochs=cfg.epochs,
        batch_size=cfg.batch,
        val_ratio=cfg.val_ratio,
        atoms_per_call=cfg.apc,
        scheduler=lr_scheduler,
        reset_early_stopping=cfg.reset_early_stopping,
        enable_shuffle=not cfg.no_shuffle,
        mini_epoch_size=cfg.mini_epoch,
        early_stopping=cfg.early_stopping,
        checkpoint_save=cfg.checkpoint_save,
        subsetpdbs=cfg.subsetpdbs,
        energy_weight=cfg.energy_weight,
        force_weight=cfg.force_weight,
        energy_matching=cfg.energy_weight != 0.0,
        train_term_def=train_term_def,
        embedding_filename=cfg.embedding,
        dataset_chunk_size=cfg.chunk_dataset,
        use_npfile=cfg.npfile,
        use_force_weights=cfg.use_force_weights,
        num_workers=cfg.num_workers,
    )


if __name__ == "__main__":
    from module.base_config import BaseConfig

    cfg = BaseConfig(build_parser())
    validate_config(cfg)
    params = process_config(cfg)

    ddp = DistributedContext()
    if cfg.distributed or 'RANK' in os.environ or 'SLURM_PROCID' in os.environ:
        ddp = setup_distributed()
        if ddp.is_distributed and ddp.is_main:
            print(f"Initialized process {ddp.rank}/{ddp.world_size} (local_rank={ddp.local_rank})")

    try:
        train_model(**params, ddp=ddp)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        sys.exit(1)
    finally:
        cleanup_distributed()
