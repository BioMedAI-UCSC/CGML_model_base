#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
# ========== DISTRIBUTED TRAINING IMPORTS - NEW ==========
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# ========================================================
import yaml
import numpy as np
# from torchmdnet.models.model import create_model
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

# Type hinting...
from typing import Tuple
from torch import Tensor

# Useful for debugging pytorch CUDA crashes
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"

# ========== DISTRIBUTED SETUP FUNCTIONS - NEW ==========
def setup_distributed():
    """Initialize the distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    else:
        print("Not using distributed mode")
        return None, None, None
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    """Check if this is the main process."""
    return rank is None or rank == 0
# ========================================================

def flatten_first(t):
    """Flatten the first two dimentions of tensor t"""
    if t is None:
        return t
    if len(t.shape) < 2:
        return t
    return t.reshape(t.shape[0]*t.shape[1], *t.shape[2:])

def make_term_offsets(lengths, term_lengths):
    result = []
    count = 0
    repeats = len(term_lengths)//len(lengths)
    lengths = np.tile(lengths, repeats)
    assert len(lengths) == len(term_lengths)
    # For each batch we want to offset the indicies used by the terms by the number of atoms in the prior batches
    for off, nterms in zip(lengths, term_lengths):
        result.append(torch.full((nterms, 1), count, dtype=torch.long))
        count += off
    return torch.cat(result)

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
        
        #TODO: It would be better if the term lengths were also python lists like the batch lengths to avoid the round trip to the GPU and back
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
        return result #pyright: ignore[reportReturnType]

class TermDef():
    def __init__(self, path=None, conf=None):
        self.scales = {}
        self.angle_wrap = {}
        if path:
            with open(path, 'r') as file:
                conf = yaml.safe_load(file)
        if conf:
            for k, v in conf.items():
                if v is not None and "scale" in v:
                    self.scales[k] = float(v["scale"])
                else:
                    self.scales[k] = 1.0
                if v is not None and "angle_wrap" in v:
                    self.angle_wrap[k] = bool(v["angle_wrap"])
                else:
                    self.angle_wrap[k] = False
    
    def get_names(self):
        return list(self.scales.keys())
    
    def get_scale(self, name):
        return self.scales[name]
    
    def get_angle_wrap(self, name):
        return self.angle_wrap[name]

def deterministic_shuffle(target, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n=len(target), generator=generator, device="cpu")
    return [target[i] for i in indices]

def check_early_stopping(val_list, patience=1):
    """Return True if the number of epochs with increasing val_loss > patience. If patience < 0 always return False."""
    if patience < 0:
        return False
    if len(val_list) < patience+2:
        return False
    check_range = np.array(val_list[-(patience+2):])
    if np.all((check_range[1:]-check_range[:-1])>0):
        print(f"Validation loss increased {patience+1} times, stopping...")
        return True

# ========== MODIFIED: Added rank parameter ==========
def save_checkpoint(checkpoint_path, epoch, model, optimizer, model_conf, scheduler, extra=None, rank=None):
    """Save checkpoint only on main process."""
    # CHANGE: Only main process saves checkpoints
    if not is_main_process(rank):
        return
    
    # CHANGE: For DDP, extract the underlying model
    if isinstance(model, DDP):
        model_state = model.module.model.state_dict()
    elif hasattr(model, 'module'):
        # DataParallel case
        model_state = model.module.model.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint_dict = {
        "epoch":epoch,
        "optimizer":optimizer.state_dict(),
        "state_dict":model_state,
        "hyper_parameters":model_conf,
    }
    if scheduler:
        checkpoint_dict["scheduler"] = scheduler.state_dict()
    if extra:
        checkpoint_dict["extra"] = extra
    
    torch.save(checkpoint_dict, checkpoint_path)
# ====================================================

# ========== MODIFIED: Added distributed parameters ==========
def gen_dataloaders(directory_path, pdb_list, energy_filename, embedding_filename, use_npfile, enable_shuffle, val_ratio, batch_size, atoms_per_call, rank=None, world_size=None):
    print("Dataset:", " ".join(pdb_list))
    all_data = dataset.ProteinDataset(directory_path, pdb_list, energy_file=energy_filename, embeddings_file=embedding_filename, use_npfile=use_npfile)
    
    # num_proteins = all_data.num_proteins()
    assert val_ratio > 0.0 and val_ratio < 1.0
    val_size = int(val_ratio * len(all_data))
    train_size = len(all_data) - val_size
    
    if enable_shuffle:
        # Generate the test and validation split with deterministic indices
        generator1 = torch.Generator().manual_seed(12341234)
        val_idx, train_idx = torch.utils.data.random_split(torch.arange(len(all_data)), [val_size, train_size], generator=generator1) #pyright: ignore[reportArgumentType]
    else:
        # Data was pre-shuffled during preprocess and can be read sequentially
        train_idx = range(train_size)
        val_idx = range(train_size, train_size+val_size)
    
    train = torch.utils.data.Subset(all_data, train_idx) #pyright: ignore[reportArgumentType]
    val = torch.utils.data.Subset(all_data, val_idx) #pyright: ignore[reportArgumentType]
    
    collate_fn = dataset.ProteinBatchCollate(atoms_per_call)
    
    # ========== CHANGE: Add DistributedSampler for DDP ==========
    train_sampler = None
    val_sampler = None
    if world_size is not None and world_size > 1:
        train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=False)
        val_sampler = DistributedSampler(val, num_replicas=world_size, rank=rank, shuffle=False)
    # ============================================================
    
    train_data = DataLoader(train, batch_size=batch_size, shuffle=(train_sampler is None and False), num_workers=4, persistent_workers=True, pin_memory=True, collate_fn=collate_fn, sampler=train_sampler)
    val_data = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True, collate_fn=collate_fn, sampler=val_sampler)
    
    # print(f"Number of proteins in the dataset: {num_proteins}")
    # print(f"Using periodic box: {all_data.has_box()}") 
    # ========== CHANGE: Return train_sampler ==========
    return all_data, train_data, val_data, train_sampler
    # ==================================================

class RoundRobinDataWrapper:
    def __init__(self, *iterables):
        self.iterables = iterables
    
    def __len__(self):
        return sum(map(len, self.iterables))
    
    def __iter__(self):
        # From https://docs.python.org/3/library/itertools.html#itertools-recipes
        iterators = map(iter, self.iterables)
        for num_active in range(len(self.iterables), 0, -1):
            iterators = itertools.cycle(itertools.islice(iterators, num_active))
            yield from map(next, iterators)

# ========== MODIFIED: Added distributed parameters ==========
def train_model(directory_path, conf_path, result_directory, dry_run, gpu_ids, weight_decay, learning_rate, epochs, batch_size, val_ratio, atoms_per_call, scheduler, reset_early_stopping, enable_shuffle, mini_epoch_size, early_stopping, checkpoint_save, subsetpdbs, energy_weight, force_weight, energy_matching, train_term_def, embedding_filename, dataset_chunk_size, use_npfile, use_force_weights, rank=None, world_size=None, local_rank=None):
    
    # ========== CHANGE: Determine if this is main process ==========
    is_main = is_main_process(rank)
    
    # CHANGE: Set device based on local_rank for DDP
    if local_rank is not None:
        device = torch.device(f'cuda:{local_rank}')
    elif gpu_ids != "cpu":
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # ================================================================
    
    with open(os.path.join(directory_path, "result", subsetpdbs), 'r') as file:
        pdb_list = file.read().split('\n')
    
    # Remove duplicates and empty strings
    pdb_list = sorted(list(set([i for i in pdb_list if i])))
    pdb_list = deterministic_shuffle(pdb_list, seed=47563537)
    
    if dataset_chunk_size is not None:
        pdb_lists = [pdb_list[i:i + dataset_chunk_size] for i in range(0, len(pdb_list), dataset_chunk_size)]
    else:
        pdb_lists = [pdb_list]
    
    # Load all proteins into a datasets
    energy_filename = None
    if energy_matching:
        energy_filename = "tica_delta_energies.npy"
    
    if embedding_filename is None:
        embedding_filename = "embeddings.npy"
    
    datasets = []
    train_dataloaders = []
    val_dataloaders = []
    # ========== CHANGE: Track samplers for epoch setting ==========
    train_samplers = []
    # ==============================================================
    
    for pdb_chunk in pdb_lists:
        # ========== CHANGE: Pass rank and world_size ==========
        ds, train_loader, val_loader, train_sampler = gen_dataloaders(directory_path, pdb_chunk, energy_filename, embedding_filename, use_npfile, enable_shuffle, val_ratio, batch_size, atoms_per_call, rank=rank, world_size=world_size)
        # ======================================================
        datasets.append(ds)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)
        train_samplers.append(train_sampler)
    
    train_data = RoundRobinDataWrapper(*train_dataloaders)
    val_data = RoundRobinDataWrapper(*val_dataloaders)
    
    # Create the model
    if conf_path is None:
        conf_path = "../configs/config.yaml"
    
    with open(conf_path, 'r') as file:
        conf = yaml.safe_load(file)
    
    # ========== CHANGE: Only main process prints ==========
    if is_main:
        print("Config:\n", conf, "\n")
    
    if conf.get("external_embedding_channels") == None and embedding_filename != "embeddings.npy":
        if is_main:
            print("WARNING: external embeddings usually should use graph-network-ext network")
    # ======================================================
    
    # Set the network to return the harmonic term info if we're training them
    if "harmonic_net" in conf and train_term_def.get_names():
        conf["harmonic_net_return_terms"] = True
    
    model = create_model(args=conf)
    
    # ========== CHANGE: Move model to device before wrapping ==========
    model.to(device)
    # ===================================================================
    
    # We need to construct DataParallel and move the model to CUDA before
    # initializing the optimizer or we get "Expected all tensors to be on the same device"
    # errors. When exactly this error happens depends on how many GPUs are used and whether
    # we're loading a checkpoint or not.
    
    # ========== CHANGE: Use DDP for distributed, DataParallel for single-node ==========
    wrapped_model = BatchWrapper(model)
    
    if world_size is not None and world_size > 1:
        # Distributed Data Parallel for multi-node/multi-GPU
        parallel_model = DDP(wrapped_model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

        device_output = device
        if is_main:
            print(f"DDP: Training on {world_size} GPUs across nodes (local_rank={local_rank})")
    elif gpu_ids == "cpu":
        parallel_model = wrapped_model
        device_output = "cpu"
        if is_main:
            print("Training on CPU")
    else:
        parallel_model = nn.DataParallel(wrapped_model, device_ids=gpu_ids)
        device_output = parallel_model.output_device
        if is_main:
            print(f"DataParallel: Training on {len(parallel_model.device_ids)} GPU(s)")
    # ====================================================================================
    
    if is_main:
        print("Model:\n", model, "\n")
    
    extra_train_terms = []
    
    # Add additional features to the dataset if the model requires them
    if "sequence_basis_radius" in conf:
        if is_main:
            print(f"Adding sequences to dataset... (sequence_basis_radius={conf['sequence_basis_radius']})")
        for d in datasets:
            d.build_sequences()
    
    if "harmonic_net" in conf:
        if is_main:
            print(f"Adding classical terms to dataset... (harmonic_net={conf['harmonic_net']})")
        for d in datasets:
            d.build_classical_terms()
        
        if train_term_def.get_names():
            # FIXME: Rename this to more generic
            harmonic_trained_terms = train_term_def.get_names()
            if is_main:
                print(f"Loading additional trained terms: {harmonic_trained_terms}")
            for d in datasets:
                d.load_frame_terms(harmonic_trained_terms)
            extra_train_terms.extend(harmonic_trained_terms)
            if is_main:
                print(f"  Term Scales: {[train_term_def.get_scale(i) for i in harmonic_trained_terms]}")
                print(f"  Term Angle Wrap: {[train_term_def.get_angle_wrap(i) for i in harmonic_trained_terms]}")
    
    if use_force_weights:
        if is_main:
            print("Loading forces weights...")
        for d in datasets:
            d.load_frame_terms(["forces_weights"])
    
    if is_main:
        print()
    
    criterion = nn.MSELoss()
    term_criterion = nn.MSELoss(reduction="none")

    # ========== DSM extension (cgff/Durumeric 2026) ==========
    # Default weight 0 = fully disabled = pure FM. Args reached via the
    # module-level `args` set in __main__.
    dsm_enabled = args.dsm_weight > 0
    if dsm_enabled:
        sys.path.insert(0, "/u/awaghili/FoundationModel")
        from cgff.extensions.dsm.schedule import GaussianNoiseSchedule
        from cgff.extensions.dsm.loss import kT_kJmol
        dsm_schedule = GaussianNoiseSchedule(
            sigma_min_nm=args.dsm_sigma_min,
            sigma_max_nm=args.dsm_sigma_max,
            n_levels=args.dsm_n_levels,
        )
        dsm_kT = kT_kJmol(args.dsm_temperature)
        dsm_weight = args.dsm_weight
        if is_main_process(rank):
            print(f"[DSM] enabled weight={dsm_weight} "
                  f"sigma=[{args.dsm_sigma_min},{args.dsm_sigma_max}] nm "
                  f"K={args.dsm_n_levels} T={args.dsm_temperature}K "
                  f"kT={dsm_kT:.4f} kJ/mol")
    # =========================================================

    do_decay = []
    dont_decay = []
    for name, param in model.named_parameters():
        if should_decay(name):
            do_decay.append(param)
        else:
            dont_decay.append(param)
    
    optimizer = optim.AdamW(
        [
            {"params": do_decay, "weight_decay": weight_decay},
            {"params": dont_decay}
        ],
        lr=learning_rate)
    
    if scheduler:
        scheduler.initialize(optimizer)
    
    epoch_resume = None
    checkpoint_path = None
    
    if os.path.exists(f'{result_directory}/checkpoint-mini.pth'):
        checkpoint_path = f'{result_directory}/checkpoint-mini.pth'
    elif os.path.exists(f'{result_directory}/checkpoint.pth'):
        checkpoint_path = f'{result_directory}/checkpoint.pth'
    
    if is_main:
        print("checkpoint_path", checkpoint_path)
    
    if checkpoint_path:
        if is_main:
            print("Resuming:", result_directory)
        
        # ========== CHANGE: Load checkpoint to correct device ==========
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        # ===============================================================
        
        # ========== CHANGE: Load into correct model wrapper ==========
        if isinstance(parallel_model, DDP):
            model_util.load_state_dict_with_rename(parallel_model.module.model, checkpoint["state_dict"])
        elif hasattr(parallel_model, 'module'):
            model_util.load_state_dict_with_rename(parallel_model.module.model, checkpoint["state_dict"])
        else:
            model_util.load_state_dict_with_rename(model, checkpoint["state_dict"])
        # =============================================================
        
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            if is_main:
                print("  No optimizer in checkpoint, resetting...")
        
        if scheduler and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        
        if "extra" in checkpoint:
            # This was a mini-checkpoint
            epoch_resume = checkpoint["extra"]
        
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
        else:
            epoch = 0
    else:
        if not result_directory or not os.path.exists(result_directory):
            if not result_directory:
                result_directory = "../data/result-" + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
            # ========== CHANGE: Only main process creates directory ==========
            if not dry_run and is_main:
                os.makedirs(result_directory, exist_ok=False)
            if is_main:
                print("Created:", result_directory)
            # =================================================================
        elif os.path.exists(f'{result_directory}/training_info.json'):
            # Most likely the training started but was canceled/crashed before the first epoch finished
            if is_main:
                print("Re-initializing:", result_directory)
        else:
            raise RuntimeError("Model directory exists but doesn't contain a checkpoint.pth or training_info.json file")
        epoch = 0
    
    # ========== CHANGE: Synchronize all processes after initialization ==========
    if world_size is not None and world_size > 1:
        dist.barrier()
    # =============================================================================
    
    epoch_history = {}
    train_loss_list = []
    val_loss_list = []
    energy_loss_list = []
    force_loss_list = []
    
    if epoch > 0:
        # Load the numpy history files
        history = np.load(f'{result_directory}/history.npy', allow_pickle=True).item()
        train_loss_list = history['train']
        val_loss_list = history['val']
        energy_loss_list = history['energy']
        force_loss_list = history['force']
        
        # Might exist before epoch 1 if mini-checkpoints were saved
        epoch_history_path = os.path.join(result_directory, "epoch_history.json")
        if os.path.exists(epoch_history_path):
            with open(epoch_history_path, "r") as f:
                epoch_history = json.load(f)
    
    if is_main:
        print("Saving to:", result_directory)
    
    # Document training parameters and input data
    # ========== CHANGE: Only main process writes files ==========
    if is_main:
        training_info_path = os.path.join(result_directory, "training_info.json")
        training_info_dict = {}
        
        if os.path.exists(training_info_path):
            with open(training_info_path, "r") as f:
                training_info_dict = json.load(f)
            
            # Check for the old dict format and update it
            if "input_directory" in training_info_dict.keys():
                training_info_dict = {"0": training_info_dict}
        else:
            print("Path", training_info_path, "does not exist")
        
        # TODO: Only add a new entry if the parameters have changed?
        training_info_dict[str(epoch)] = {
            "weight_decay" : weight_decay,
            "learning_rate" : learning_rate,
            "epochs" : epochs,
            "batch_size" : batch_size,
            "input_directory" : directory_path,
            "pdbs" : pdb_list,
            "energy_weight": energy_weight,
            "force_weight": force_weight,
            "embedding_filename" : embedding_filename,
            "world_size": world_size if world_size else 1,  # CHANGE: Track world_size
        }
        
        if scheduler:
            training_info_dict[str(epoch)]["lr_scheduler"] = repr(scheduler)
        else:
            # If there's no scheduler reset the learning of the optimizer to the passed value
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        
        if not dry_run:
            with open(training_info_path, "w") as f:
                json.dump(training_info_dict, f, indent=2)
        
        # Save the validation frame indices
        # FIXME: This isn't compatible with chunking
        #np.save(os.path.join(result_directory, "validation_frames.npy"), np.array(val_idx))
        
        # Save the prior with the model
        prior_path = os.path.join(directory_path, "priors.yaml")
        if os.path.exists(prior_path):
            prior_params_path = os.path.join(directory_path, "prior_params.json")
            shutil.copy(prior_path, result_directory)
            shutil.copy(prior_params_path, result_directory)
    # ============================================================
    
    # Disable earlly stopping when using an annealing (cycling) schedualer
    if scheduler and scheduler.is_annealing():
        early_stopping = -1
    
    first_early_stopping_epoch = 0
    if reset_early_stopping == True:
        first_early_stopping_epoch = epoch
    
    verbose_loss_report = sys.stdout.isatty()
    
    while epoch < epochs:
        # ========== CHANGE: Set epoch for distributed samplers ==========
        if world_size is not None and world_size > 1:
            for sampler in train_samplers:
                if sampler is not None:
                    sampler.set_epoch(epoch)
        # ================================================================
        
        t0 = time.time()
        model.train()
        
        train_loss = 0
        train_energy_loss = 0
        train_force_loss = 0
        num_cal = 0  # The total number of elements trained on
        epoch_offset = 0
        mini_train_loss = 0
        mini_num_cal = 0
        
        train_term_losses = {k: 0.0 for k in extra_train_terms}
        train_term_num_cal = {k: 0 for k in extra_train_terms}
        
        if epoch_resume:
            if is_main:
                print("Resuming epoch...")
            train_loss = float(epoch_resume["train_loss"])
            num_cal = int(epoch_resume["num_cal"])
            epoch_offset = int(epoch_resume["i"])
            epoch_resume = None
        
        # ========== CHANGE: Only show progress bar on main process ==========
        # Setting miniters is required to keep the bar from stalling after skipping ahead while resuming a batch
        if is_main:
            tqdm_iter = tqdm(enumerate(train_data), desc=f"Training ({epoch}/{epochs})", total=len(train_data), dynamic_ncols=True, miniters=1)
        else:
            tqdm_iter = enumerate(train_data)
        # ====================================================================
        
        for i, batch in tqdm_iter:
            # Handle mini-batches
            if i < epoch_offset:
                # TODO: It's very wasteful to load everything then discard it, but the alternative requires making a 2nd dataset object...
                continue
            elif epoch_offset and i == epoch_offset:
                if is_main and hasattr(tqdm_iter, 'write'):
                    tqdm_iter.write(f"Resumed epoch at batch {i}")
            elif mini_epoch_size and 0 == i % mini_epoch_size and i > 0:
                tmp_checkpoint_path = f'{result_directory}/checkpoint-{epoch}-{i}.pth'
                # ========== CHANGE: Pass rank to save_checkpoint ==========
                save_checkpoint(tmp_checkpoint_path, epoch, parallel_model if isinstance(parallel_model, DDP) else model, optimizer, conf, scheduler, extra = {"train_loss":train_loss, "num_cal":num_cal, "i":i}, rank=rank)
                # ===========================================================
                
                if is_main:
                    os.replace(tmp_checkpoint_path, f'{result_directory}/checkpoint-mini.pth')
                    
                    epoch_history[f"{epoch}-{i}"] = {
                        "train_loss":train_loss/num_cal,
                        "mini_train_loss":mini_train_loss/mini_num_cal,
                        "epoch_len":len(train_data),
                        "lr":[g['lr'] for g in optimizer.param_groups],
                    }
                    
                    epoch_history_path = os.path.join(result_directory, "epoch_history.json")
                    with open(epoch_history_path, "w") as f:
                        json.dump(epoch_history, f, indent=2)
                    
                    if hasattr(tqdm_iter, 'write'):
                        tqdm_iter.write(f"Mini-epoch {epoch}-{i}: Train Loss: {train_loss/num_cal}")
                
                mini_train_loss = 0
                mini_num_cal = 0
            
            total_batch_size = sum([i["force"].numel() for i in batch])
            num_cal += total_batch_size
            mini_num_cal += total_batch_size
            
            total_term_batch_size = {k: sum([i[k].numel() for i in batch]) for k in train_term_losses}
            for k in train_term_num_cal.keys():
                train_term_num_cal[k] += total_term_batch_size[k]
            
            optimizer.zero_grad()
            
            for sub_batch in batch:
                force = sub_batch.pop("force")
                force = force.reshape(-1, force.shape[-1]).to(device_output)
                
                force_weights = None
                if use_force_weights:
                    force_weights = sub_batch.pop("forces_weights")
                    force_weights = force_weights.reshape(-1).to(device_output)
                
                energy = None
                if energy_matching:
                    energy = sub_batch.pop("energy")
                    energy = energy.reshape(-1, energy.shape[-1]).to(device_output)
                
                term_targets = {}
                for k in train_term_losses.keys():
                    term_targets[k] = sub_batch.pop(k).flatten().to(device_output)
                
                out_energy, out_force, extra = parallel_model(**sub_batch)
                
                # Scale the sub_batch to be a term in the overall mean of the batch
                sub_batch_size = force.numel()
                energy_loss: torch.Tensor = torch.tensor(0.0)
                
                if energy_matching:
                    energy_loss = criterion(out_energy, energy) * (sub_batch_size / total_batch_size)
                
                force_loss = term_criterion(out_force, force)
                if force_weights is not None:
                    force_loss = force_loss*force_weights[:, None]
                force_loss = force_loss.mean() * (sub_batch_size / total_batch_size)

                loss = energy_weight * energy_loss + force_weight * force_loss

                # ========== DSM extension (cgff/Durumeric 2026) ==========
                # Additive Karras-weighted denoising-score-matching loss on a
                # per-sub-batch noisy forward.
                if dsm_enabled:
                    pos_cur = sub_batch["pos"]
                    B = pos_cur.shape[0]
                    sigma_b = dsm_schedule.sample(B).to(pos_cur.device, pos_cur.dtype)
                    sigma_bcast = sigma_b.view(B, 1, 1)
                    noise = torch.randn(pos_cur.shape, device=pos_cur.device,
                                        dtype=pos_cur.dtype) * sigma_bcast
                    noisy_batch = dict(sub_batch)
                    noisy_batch["pos"] = pos_cur + noise
                    out_noisy = parallel_model(**noisy_batch)
                    noisy_force = out_noisy[1] if isinstance(out_noisy, (tuple, list)) else out_noisy["out_force"]
                    dev = noisy_force.device
                    N_per = pos_cur.shape[1]
                    sigma_per_node = sigma_b.to(dev).repeat_interleave(N_per).view(-1, 1)
                    target = -dsm_kT * noise.reshape(-1, 3).to(dev) / (sigma_per_node ** 2).clamp_min(1e-12)
                    diff = noisy_force - target
                    sq = diff.pow(2).sum(dim=-1)
                    weight_per_node = (sigma_per_node.squeeze(-1) ** 2)
                    dsm_term = dsm_weight * (weight_per_node * sq).mean() * (sub_batch_size / total_batch_size)
                    loss = loss + dsm_term
                # =========================================================

                train_force_loss += force_loss.item() * total_batch_size
                if energy_matching:
                    train_energy_loss += (energy_loss.item() * total_batch_size)

                delta_loss = loss.item() * total_batch_size
                train_loss += delta_loss
                mini_train_loss += delta_loss

                for k in train_term_losses.keys():
                    # TODO: Find a more generic way of doing this
                    if train_term_def.get_angle_wrap(k):
                        train_term_loss = (extra[k] - term_targets[k] + torch.pi) % (2*torch.pi) - torch.pi
                        train_term_loss = train_term_loss**2
                    else:
                        train_term_loss = term_criterion(extra[k], term_targets[k])
                    
                    # We don't multiply by numel here because the term loss criterion doesn't do a mean reduction
                    train_term_loss = train_term_loss / total_term_batch_size[k]
                    # Mask out undefined values
                    # TODO: Ensure this is a good threshold value
                    train_term_loss = train_term_loss * (term_targets[k] >= -10).float()
                    train_term_loss = torch.sum(train_term_loss)
                    loss = loss + train_term_loss*train_term_def.get_scale(k)
                    train_term_losses[k] += train_term_loss.item() * total_term_batch_size[k]
                
                # Accumulate gradient
                loss.backward()
            
            optimizer.step()
            
            if scheduler:
                scheduler.step_batch(epoch + i/len(train_data))
            
            if dry_run:
                if is_main:
                    print("\nDry run OK!")
                sys.exit(0)
            
            if verbose_loss_report and is_main and hasattr(tqdm_iter, 'set_description'):
                desc = [f"Training ({epoch}/{epochs}) (T={train_loss/num_cal:.4f}"]
                for t in train_term_losses:
                    desc.append(f"{t}={train_term_losses[t]/train_term_num_cal[t]:.4f}")
                desc = ", ".join(desc) + ")"
                tqdm_iter.set_description(desc)
        
        # ========== CHANGE: Aggregate training losses across all processes ==========
        if world_size is not None and world_size > 1:
            train_loss_tensor = torch.tensor(train_loss, device=device)
            num_cal_tensor = torch.tensor(num_cal, device=device)
            train_energy_loss_tensor = torch.tensor(train_energy_loss, device=device)
            train_force_loss_tensor = torch.tensor(train_force_loss, device=device)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_cal_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_energy_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_force_loss_tensor, op=dist.ReduceOp.SUM)
            
            train_loss = train_loss_tensor.item()
            num_cal = num_cal_tensor.item()
            train_energy_loss = train_energy_loss_tensor.item()
            train_force_loss = train_force_loss_tensor.item()
            
            # Aggregate term losses
            for k in train_term_losses.keys():
                term_loss_tensor = torch.tensor(train_term_losses[k], device=device)
                term_num_cal_tensor = torch.tensor(train_term_num_cal[k], device=device)
                dist.all_reduce(term_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(term_num_cal_tensor, op=dist.ReduceOp.SUM)
                train_term_losses[k] = term_loss_tensor.item()
                train_term_num_cal[k] = term_num_cal_tensor.item()
        # =============================================================================
        
        train_loss_list.append(train_loss/num_cal)
        energy_loss_list.append(train_energy_loss/num_cal)
        force_loss_list.append(train_force_loss/num_cal)
        
        parallel_model.eval()
        val_loss = 0
        num_cal = 0
        
        val_term_losses = {k: 0.0 for k in train_term_losses}
        val_term_num_cal = {k: 0 for k in train_term_num_cal}
        
        # ========== CHANGE: Only show progress bar on main process ==========
        if is_main:
            val_iter = tqdm(val_data, desc=f"Validation ({epoch}/{epochs})", total=len(val_data), dynamic_ncols=True)
        else:
            val_iter = val_data
        # ====================================================================
        
        for batch in val_iter:
            total_batch_size = sum([i["force"].numel() for i in batch])
            num_cal += total_batch_size
            
            total_term_batch_size = {k: sum([i[k].numel() for i in batch]) for k in val_term_losses}
            for k in val_term_num_cal.keys():
                val_term_num_cal[k] += total_term_batch_size[k]
            
            for sub_batch in batch:
                force = sub_batch.pop("force")
                force = force.reshape(-1, force.shape[-1]).to(device_output)
                
                force_weights = None
                if use_force_weights:
                    force_weights = sub_batch.pop("forces_weights")
                    force_weights = force_weights.reshape(-1).to(device_output)
                
                energy = None
                if energy_matching:
                    energy = sub_batch.pop("energy")
                    energy = energy.reshape(-1, energy.shape[-1]).to(device_output)
                
                term_targets = {}
                for k in val_term_losses.keys():
                    term_targets[k] = sub_batch.pop(k).flatten().to(device_output)
                
                out_energy, out_force, extra = parallel_model(**sub_batch)
                
                sub_batch_size = force.numel()
                energy_loss: torch.Tensor = torch.tensor(0.0)
                
                if energy_matching:
                    energy_loss = criterion(out_energy, energy) * (sub_batch_size / total_batch_size)
                
                force_loss = term_criterion(out_force, force)
                if force_weights is not None:
                    force_loss = force_loss*force_weights[:, None]
                force_loss = force_loss.mean() * (sub_batch_size / total_batch_size)
                
                loss = energy_weight * energy_loss + force_weight * force_loss
                val_loss += loss.item() * total_batch_size
                
                for k in val_term_losses.keys():
                    if train_term_def.get_angle_wrap(k):
                        val_term_loss = (extra[k] - term_targets[k] + torch.pi) % (2*torch.pi) - torch.pi
                        val_term_loss = val_term_loss**2
                    else:
                        val_term_loss = term_criterion(extra[k], term_targets[k])
                    
                    val_term_loss = val_term_loss / total_term_batch_size[k]
                    val_term_loss = val_term_loss * (term_targets[k] >= -10).float()
                    val_term_loss = torch.sum(val_term_loss)
                    # loss = loss + val_term_loss*term_val_weight
                    val_term_losses[k] += val_term_loss.item() * total_term_batch_size[k]
    
        # ========== CHANGE: Aggregate validation losses across all processes ==========
        if world_size is not None and world_size > 1:
            val_loss_tensor = torch.tensor(val_loss, device=device)
            num_cal_tensor = torch.tensor(num_cal, device=device)
            
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_cal_tensor, op=dist.ReduceOp.SUM)
            
            val_loss = val_loss_tensor.item()
            num_cal = num_cal_tensor.item()
            
            # Aggregate term losses
            for k in val_term_losses.keys():
                term_loss_tensor = torch.tensor(val_term_losses[k], device=device)
                term_num_cal_tensor = torch.tensor(val_term_num_cal[k], device=device)
                dist.all_reduce(term_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(term_num_cal_tensor, op=dist.ReduceOp.SUM)
                val_term_losses[k] = term_loss_tensor.item()
                val_term_num_cal[k] = term_num_cal_tensor.item()
        # ===============================================================================
        
        val_loss_list.append(val_loss/num_cal)
        
        if scheduler:
            scheduler.step(val_loss/num_cal)
        
        # ========== CHANGE: Only main process saves epoch history ==========
        if is_main:
            epoch_history_path = os.path.join(result_directory, "epoch_history.json")
            epoch_history[f"{epoch}"] = {
                "train_loss":train_loss_list[-1],
                "val_loss":val_loss_list[-1],
                "energy_loss":energy_loss_list[-1],
                "force_loss":force_loss_list[-1],
                "epoch_len":len(train_data),
                "lr":[g['lr'] for g in optimizer.param_groups],
            }
            
            for k in extra_train_terms:
                epoch_history[f"{epoch}"][f"train_loss_{k}"] = train_term_losses[k]/train_term_num_cal[k]
                epoch_history[f"{epoch}"][f"val_loss_{k}"] = val_term_losses[k]/val_term_num_cal[k]
            
            with open(epoch_history_path, "w") as f:
                json.dump(epoch_history, f, indent=2)
            
            print(f"Epoch {epoch} - Train Loss: {train_loss_list[-1]} - Val Loss: {val_loss_list[-1]} - time: {round(time.time() - t0,2)}s")
            if epoch > 0:
                print(f"  ∆Train: {train_loss_list[-1]-train_loss_list[-2]} - ∆Val: {val_loss_list[-1] - val_loss_list[-2]}")
                # print(f"  ∆Energy: {energy_loss_list[-1]-energy_loss_list[-2]} - ∆Force: {force_loss_list[-1] - force_loss_list[-2]}")
            
            for k in val_term_losses.keys():
                print(f"  Train {k} loss={train_term_losses[k]/train_term_num_cal[k]:.4f}")
                print(f"  Val {k} loss={val_term_losses[k]/val_term_num_cal[k]:.4f}")
        # ===================================================================
        
        # ========== CHANGE: Check early stopping on main, broadcast decision ==========
        should_stop = False
        if is_main:
            should_stop = check_early_stopping(val_loss_list[first_early_stopping_epoch:], patience=early_stopping)
        
        # Broadcast early stopping decision to all processes
        if world_size is not None and world_size > 1:
            should_stop_tensor = torch.tensor(1 if should_stop else 0, device=device)
            dist.broadcast(should_stop_tensor, src=0)
            should_stop = bool(should_stop_tensor.item())
        
        if should_stop:
            if is_main:
                print("Early stopping triggered.")
            break
        # ===============================================================================
        
        history = {"train": train_loss_list, "val": val_loss_list, "energy": energy_loss_list, "force": force_loss_list}
        
        # Save the model
        # I've attempted to make this compatible with the TorchMD calculators.External class, but I'm not sure how well the keys match - Daniel
        tmp_checkpoint_path = f'{result_directory}/checkpoint-{epoch}.pth'
        # ========== CHANGE: Pass rank to save_checkpoint ==========
        save_checkpoint(tmp_checkpoint_path, epoch + 1, parallel_model if isinstance(parallel_model, DDP) else model, optimizer, conf, scheduler, rank=rank)
        # ===========================================================
        
        # ========== CHANGE: Only main process manages checkpoint files ==========
        if is_main:
            if os.path.exists(f'{result_directory}/checkpoint-mini.pth'):
                os.unlink(f'{result_directory}/checkpoint-mini.pth')
            
            if checkpoint_save and (epoch % checkpoint_save == 0):
                shutil.copyfile(tmp_checkpoint_path, f'{result_directory}/checkpoint.pth')
            else:
                os.replace(tmp_checkpoint_path, f'{result_directory}/checkpoint.pth')
            
            # If this is <= to the lowest validation loss seen so far also save it to checkpoint-best.pth
            if val_loss_list[-1] <= np.min(val_loss_list):
                shutil.copyfile(f'{result_directory}/checkpoint.pth', f'{result_directory}/checkpoint-best.pth')
            
            # Save the loss history
            np.save(f'{result_directory}/history.npy', history)#pyright: ignore[reportArgumentType]
            print("  Checkpoint saved.")
        # =========================================================================
        
        # ========== CHANGE: Synchronize all processes before next epoch ==========
        if world_size is not None and world_size > 1:
            dist.barrier()
        # =========================================================================
        
        epoch += 1

def should_decay(param_name: str) -> bool:
    #usually something like "representation_model.distance_expansion.means"
    #want to not decay the embeddings and the biases
    parts = param_name.split('.')
    assert len(parts) > 0
    if parts[-1] == "bias":
        return False
    if parts[-2] == "embedding":
        return False
    if parts[-2] == "distance_expansion":
        #not sure for this
        return False
    assert parts[-1] == "weight"
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a CGSchNet network with distributed support")
    parser.add_argument("input", help="Processed data to train on ")
    parser.add_argument("result", default=None, nargs="?", help="Checkpoint directory to continue")
    parser.add_argument("-c", "--config", default="../configs/config.yaml", type=str, help="")
    parser.add_argument("--gpus", default=None, type=str, help="List of GPUs to train on (e.g. \"0,1,2\") or 'cpu'")
    parser.add_argument("--batch", type=int, default=50, help="The batch size to use")
    parser.add_argument("--epochs", type=int, default=25, help="The total number of epochs to train for")
    parser.add_argument("--lr", type=float, default="1e-4", help="Learning rate")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio, should be between 0.0 and 1.0")
    parser.add_argument("--apc", "--atoms-per-call", type=int, default=None, help="Number of atoms to include in each sub-batch")
    parser.add_argument("--cos-anneal", default=None, help="Train using cosine annealing, parameters are \"T_0,T_mult\"")
    parser.add_argument("--cos-lr", default=None, help="Train using a cosine learning rate, parameters are \"T_max,eta_min\"")
    parser.add_argument("--exp-lr", default=None, help="Train using a exponential learning rate, parameters are \"gamma\"")
    parser.add_argument("--plateau-lr", default=None, help="Train using a plateau learning rate, parameters are \"factor\" \"patience\" \"min_lr\"")
    parser.add_argument("--dry-run", action="store_true", help="Do a dry run of the training loop but produce no output")
    parser.add_argument("--reset-early-stopping", action="store_true", help="Reset the early stopping check to start from the current epoch")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the training dataset")
    parser.add_argument("--mini-epoch", type=int, default=None, help="Save a mini epoch after every n batches")
    parser.add_argument("--early-stopping", type=int, default=1, help="The number of epochs validation loss can increase before triggering early stopping or -1 to disable early stopping (default=1)")
    parser.add_argument("--checkpoint-save", type=int, default=10, help="Save a backup checkpoint every n epochs, 0 to disable (default=10)")
    parser.add_argument("--subsetpdbs", default='ok_list.txt', type=str, help="Change the pdbid list used when reading in the dataset (default=ok_list.txt)")
    parser.add_argument("--energy-weight", default=0.0, type=float, help="Energy Weighting for Loss Function")
    parser.add_argument("--force-weight", default=1.0, type=float, help="Force Weighting for Loss Function")
    parser.add_argument("--term-def", default=None, type=str, help="The path to a term definition yaml file, which can additional loss terms used during training.")
    parser.add_argument("--embedding", type=str, default=None, help="Specify an alternate file to load embeddings from (default: embeddings.npy).")
    parser.add_argument("--chunk-dataset", type=int, default=None, help="Break the dataset into chunks of n proteins per batch")
    parser.add_argument("--npfile", action="store_true", help="Use file loader instead of mmap to load dataset")
    parser.add_argument("--use-force-weights", default=False, action="store_true", help="Use per bead force weights in training")

    # cgff DSM extension (Durumeric 2026). Default weight 0 = fully disabled = pure FM.
    parser.add_argument("--dsm-weight", type=float, default=0.0, help="Additive DSM loss weight (0 = disabled).")
    parser.add_argument("--dsm-sigma-min", type=float, default=0.01, help="DSM min sigma in nm.")
    parser.add_argument("--dsm-sigma-max", type=float, default=1.0, help="DSM max sigma in nm.")
    parser.add_argument("--dsm-n-levels", type=int, default=6, help="DSM log-spaced noise levels.")
    parser.add_argument("--dsm-temperature", type=float, default=300.0, help="kT used for DSM target scaling (K).")

    # ========== NEW: Distributed training arguments ==========
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    # =========================================================
    # cgml_run additive knobs. --precision currently accepts the flag for
    # schema parity; the bf16 autocast wrap lands in a follow-up patch.
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                        help="Autocast precision (today fp32 only; bf16 falls back with warning).")
    parser.add_argument("--base-checkpoint", default=None,
                        help="Path to a checkpoint .pth to seed result_directory if empty.")

    assert torch.cuda.is_available(), "CUDA is not available, please run on a machine with CUDA or use --gpus cpu"

    args = parser.parse_args()
    if args.precision == "bf16":
        print("[train_distributed] WARNING: --precision bf16 requested but "
              "autocast wrap is not yet enabled; falling back to fp32.", flush=True)
    
    # ========== NEW: Setup distributed training ==========
    rank = None
    world_size = None
    local_rank = None
    
    if args.distributed or 'RANK' in os.environ or 'SLURM_PROCID' in os.environ:
        rank, world_size, local_rank = setup_distributed()
        if rank is not None:
            print(f"Initialized process {rank}/{world_size} (local_rank={local_rank})")
    # =====================================================
    
    directory_path = args.input
    assert os.path.isdir(directory_path), f"Input directory does not exist: {directory_path}"

    result_directory = args.result
    # Seed result_directory with --base-checkpoint if provided and dir is empty,
    # so train_distributed.py's existing resume-from-result-dir logic picks it up.
    if args.base_checkpoint is not None and result_directory is not None:
        os.makedirs(result_directory, exist_ok=True)
        if not any(name.endswith(".pth") for name in os.listdir(result_directory)):
            src = os.path.abspath(args.base_checkpoint)
            dst = os.path.join(result_directory, "checkpoint.pth")
            if is_main_process(rank):
                print(f"[train_distributed] seeding {dst} <- {src} (--base-checkpoint)", flush=True)
                shutil.copy2(src, dst)
            if dist.is_initialized():
                dist.barrier()
    conf_path = args.config
    assert os.path.isfile(conf_path), f"Config file does not exist: {conf_path}"
    
    weight_decay = args.wd
    learning_rate = args.lr
    
    if args.gpus:
        if args.gpus == "cpu":
            gpu_ids = "cpu"
        else:
            gpu_ids = [int(i) for i in args.gpus.strip().split(",")]
    else:
        gpu_ids = "cpu"
    
    epochs = args.epochs
    batch_size = args.batch
    val_ratio = args.val_ratio
    atoms_per_call = args.apc
    dry_run = args.dry_run
    reset_early_stopping = args.reset_early_stopping
    enable_shuffle = not args.no_shuffle
    mini_epoch_size = args.mini_epoch
    early_stopping = args.early_stopping
    checkpoint_save = args.checkpoint_save
    assert checkpoint_save >= 0
    subsetpdbs = args.subsetpdbs
    energy_weight = args.energy_weight
    force_weight = args.force_weight
    energy_matching = args.energy_weight != 0.0
    embedding_filename = args.embedding
    dataset_chunk_size = args.chunk_dataset
    use_npfile = args.npfile
    use_force_weights = args.use_force_weights
    
    # Relax the maximum number of open files as much as possible
    # We will potentially open a lot of files (~4 per molecule per ProteinDataset object)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    
    lr_scheduler = None
    if args.cos_anneal:
        T_0, T_mult = [int(i) for i in args.cos_anneal.split(",")]
        lr_scheduler = SchedulerWrapper_CosineAnnealingWarmRestarts(T_0, T_mult)
    if args.cos_lr:
        assert lr_scheduler is None
        T_max, eta_min = args.cos_lr.split(",")
        T_max, eta_min = int(T_max), float(eta_min)
        lr_scheduler = SchedulerWrapper_CosineAnnealingLR(T_max, eta_min)
    if args.exp_lr:
        assert lr_scheduler is None
        lr_scheduler = SchedulerWrapper_ExponentialLR(float(args.exp_lr))
    if args.plateau_lr:
        assert lr_scheduler is None
        factor, patience, threshold, min_lr = args.plateau_lr.split(",")
        factor, patience, threshold, min_lr = float(factor), int(patience), float(threshold), float(min_lr)
        lr_scheduler = SchedulerWrapper_ReduceLROnPlateau(factor, patience, threshold, min_lr)
    
    if args.term_def is not None:
        train_term_def = TermDef(path=args.term_def)
    else:
        train_term_def = TermDef()
    
    try:
        # ========== CHANGE: Pass distributed parameters ==========
        train_model(directory_path, result_directory=result_directory, conf_path=conf_path,
                   dry_run=dry_run, weight_decay=weight_decay, learning_rate=learning_rate,
                   gpu_ids=gpu_ids, epochs=epochs, batch_size=batch_size, val_ratio=val_ratio,
                   scheduler=lr_scheduler, atoms_per_call=atoms_per_call,
                   reset_early_stopping=reset_early_stopping, enable_shuffle=enable_shuffle,
                   mini_epoch_size=mini_epoch_size, early_stopping=early_stopping,
                   checkpoint_save=checkpoint_save, subsetpdbs=subsetpdbs,
                   energy_weight=energy_weight, force_weight=force_weight,
                   energy_matching=energy_matching, train_term_def=train_term_def,
                   embedding_filename=embedding_filename, dataset_chunk_size=dataset_chunk_size,
                   use_npfile=use_npfile, use_force_weights=use_force_weights,
                   rank=rank, world_size=world_size, local_rank=local_rank)
        # =========================================================
    except Exception as e:
        # Uncaught exceptions cause pytorch to hang for quite a while before exiting
        traceback.print_tb(e.__traceback__)
        print(e)
        sys.exit(1)
    finally:
        # ========== NEW: Cleanup distributed ==========
        cleanup_distributed()
        # ==============================================
