#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
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

def save_checkpoint(checkpoint_path, epoch, model, optimizer, model_conf, scheduler, extra=None):
    checkpoint_dict = {
        "epoch":epoch,
        "optimizer":optimizer.state_dict(),
        "state_dict":model.state_dict(),
        "hyper_parameters":model_conf,
        }

    if scheduler:
        checkpoint_dict["scheduler"] = scheduler.state_dict()

    if extra:
        checkpoint_dict["extra"] = extra

    torch.save(checkpoint_dict, checkpoint_path)

def gen_dataloaders(directory_path, pdb_list, energy_filename, embedding_filename, use_npfile, enable_shuffle, val_ratio, batch_size, atoms_per_call):
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

    train_data = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=4,
                            persistent_workers=True, pin_memory=True, collate_fn=collate_fn)
    val_data = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4,
                          persistent_workers=True, pin_memory=True, collate_fn=collate_fn)

    # print(f"Number of proteins in the dataset: {num_proteins}")
    # print(f"Using periodic box: {all_data.has_box()}")

    return all_data, train_data, val_data

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

def train_model(directory_path, conf_path, result_directory, dry_run, gpu_ids,
                weight_decay, learning_rate, epochs, batch_size, val_ratio, atoms_per_call,
                scheduler, reset_early_stopping, enable_shuffle, mini_epoch_size, early_stopping,
                checkpoint_save, subsetpdbs, energy_weight, force_weight, energy_matching, train_term_def,
                embedding_filename, dataset_chunk_size, use_npfile):

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

    for pdb_chunk in pdb_lists:
        ds, train_loader, val_loader = gen_dataloaders(directory_path, pdb_chunk, energy_filename, embedding_filename, use_npfile, enable_shuffle, val_ratio, batch_size, atoms_per_call)
        datasets.append(ds)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)

    train_data = RoundRobinDataWrapper(*train_dataloaders)
    val_data = RoundRobinDataWrapper(*val_dataloaders)

    # Create the model
    if conf_path is None:
        conf_path = "../configs/config.yaml"
    with open(conf_path, 'r') as file:
        conf = yaml.safe_load(file)
    print("Config:\n", conf, "\n")

    if conf.get("external_embedding_channels") == None and embedding_filename != "embeddings.npy":
        print("WARNING: external embeddings usually should use graph-network-ext network")

    # Set the network to return the harmonic term info if we're training them
    if "harmonic_net" in conf and train_term_def.get_names():
        conf["harmonic_net_return_terms"] = True

    model = create_model(args=conf)
    # We need to construct DataParallel and move the model to CUDA before
    # initializing the optimizer or we get "Expected all tensors to be on the same device"
    # errors. When exactly this error happens depends on how many GPUs are used and whether
    # we're loading a checkpoint or not.
    if gpu_ids == "cpu":
        parallel_model = BatchWrapper(model)
        device_src = "cpu"
        device_output = "cpu"
        print("Training on CPU")
    else:
        parallel_model = nn.DataParallel(BatchWrapper(model), device_ids=gpu_ids)
        device_src = parallel_model.src_device_obj
        device_output = parallel_model.output_device
        print(f"DataParallel: Training on {len(parallel_model.device_ids)} GPU(s)")

    model.to(device_src)
    print("Model:\n", model, "\n")

    extra_train_terms = []

    # Add additional features to the dataset if the model requires them
    if "sequence_basis_radius" in conf:
        print(f"Adding sequences to dataset... (sequence_basis_radius={conf['sequence_basis_radius']})")
        for d in datasets:
            d.build_sequences()

    if "harmonic_net" in conf:
        print(f"Adding classical terms to dataset... (harmonic_net={conf['harmonic_net']})")
        for d in datasets:
            d.build_classical_terms()

    if train_term_def.get_names():
        # FIXME: Rename this to more generic
        harmonic_trained_terms = train_term_def.get_names()
        print(f"Loading additional trained terms: {harmonic_trained_terms}")
        for d in datasets:
            d.load_frame_terms(harmonic_trained_terms)
        extra_train_terms.extend(harmonic_trained_terms)
        print(f"    Term Scales:     {[train_term_def.get_scale(i) for i in harmonic_trained_terms]}")
        print(f"    Term Angle Wrap: {[train_term_def.get_angle_wrap(i) for i in harmonic_trained_terms]}")

    print()

    criterion = nn.MSELoss()
    term_criterion = nn.MSELoss(reduction="none")

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

    print("checkpoint_path", checkpoint_path)
    if checkpoint_path:
        print("Resuming:", result_directory)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device_src)
        model_util.load_state_dict_with_rename(model, checkpoint["state_dict"])

        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("  No optimizer in checkpoint, resetting...")

        if scheduler and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        if "extra" in checkpoint: # This was a mini-checkpoint
            epoch_resume = checkpoint["extra"]

        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
        else:
            epoch = 0
    else:
        if not result_directory or not os.path.exists(result_directory):
            if not result_directory:
                result_directory = "../data/result-" + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
            if not dry_run:
                os.makedirs(result_directory, exist_ok=False)
            else:
                assert os.path.exists(result_directory) == False, "Result directory exists but is invalid"
            print("Created:", result_directory)
        elif os.path.exists(f'{result_directory}/training_info.json'):
            # Most likely the training started but was canceled/crashed before the first epoch finished
            print("Re-initializing:", result_directory)
        else:
            raise RuntimeError("Model directory exists but doesn't contain a checkpoint.pth or training_info.json file")
        epoch = 0

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

    print("Saving to:", result_directory)

    # Document training parameters and input data
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

    # Disable earlly stopping when using an annealing (cycling) schedualer
    if scheduler and scheduler.is_annealing():
        early_stopping = -1

    first_early_stopping_epoch = 0
    if reset_early_stopping == True:
        first_early_stopping_epoch = epoch

    verbose_loss_report = sys.stdout.isatty()

    while epoch < epochs:
        t0 = time.time()
        model.train()
        train_loss = 0
        train_energy_loss = 0
        train_force_loss = 0
        num_cal = 0 # The total number of elements trained on
        epoch_offset = 0
        mini_train_loss = 0
        mini_num_cal = 0

        train_term_losses = {k: 0.0 for k in extra_train_terms}
        train_term_num_cal = {k: 0 for k in extra_train_terms}

        if epoch_resume:
            print("Resuming epoch...")
            train_loss = float(epoch_resume["train_loss"])
            num_cal = int(epoch_resume["num_cal"])
            epoch_offset = int(epoch_resume["i"])
            epoch_resume = None


        # Setting miniters is required to keep the bar from stalling after skipping ahead while resuming a batch
        tqdm_iter = tqdm(enumerate(train_data), desc=f"Training ({epoch}/{epochs})", total=len(train_data), dynamic_ncols=True, miniters=1)
        for i, batch in tqdm_iter:
            # Handle mini-batches
            if i < epoch_offset:
                # TODO: It's very wasteful to load everything then discard it, but the alternative requires making a 2nd dataset object...
                continue
            elif epoch_offset and i == epoch_offset:
                tqdm_iter.write(f"Resumed epoch at batch {i}")
            elif mini_epoch_size and 0 == i % mini_epoch_size and i > 0:
                tmp_checkpoint_path = f'{result_directory}/checkpoint-{epoch}-{i}.pth'
                save_checkpoint(tmp_checkpoint_path, epoch, model, optimizer, conf, scheduler, extra = {"train_loss":train_loss, "num_cal":num_cal, "i":i})
                os.replace(tmp_checkpoint_path, f'{result_directory}/checkpoint-mini.pth')

                epoch_history[f"{epoch}-{i}"] = {
                    "train_loss":train_loss/num_cal,
                    "mini_train_loss":mini_train_loss/mini_num_cal,
                    "epoch_len":len(train_data),
                    "lr":[g['lr'] for g in optimizer.param_groups],
                    }
                with open(epoch_history_path, "w") as f:
                    json.dump(epoch_history, f, indent=2)
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
                force_loss = criterion(out_force, force) * (sub_batch_size / total_batch_size)
                loss = energy_weight * energy_loss + force_weight * force_loss

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
                print("\nDry run OK!")
                sys.exit(0)

            if verbose_loss_report:
                desc = [f"Training ({epoch}/{epochs}) (T={train_loss/num_cal:.4f}"]
                for t in train_term_losses:
                    desc.append(f"{t}={train_term_losses[t]/train_term_num_cal[t]:.4f}")
                desc = ", ".join(desc) + ")"
                tqdm_iter.set_description(desc)


        train_loss_list.append(train_loss/num_cal)
        energy_loss_list.append(train_energy_loss/num_cal)
        force_loss_list.append(train_force_loss/num_cal)

        parallel_model.eval()
        val_loss = 0
        num_cal = 0

        val_term_losses = {k: 0.0 for k in train_term_losses}
        val_term_num_cal = {k: 0 for k in train_term_num_cal}

        for batch in tqdm(val_data, desc=f"Validation ({epoch}/{epochs})", total=len(val_data), dynamic_ncols=True):
            total_batch_size = sum([i["force"].numel() for i in batch])
            num_cal += total_batch_size

            total_term_batch_size = {k: sum([i[k].numel() for i in batch]) for k in val_term_losses}
            for k in val_term_num_cal.keys():
                val_term_num_cal[k] += total_term_batch_size[k]

            for sub_batch in batch:
                force = sub_batch.pop("force")
                force = force.reshape(-1, force.shape[-1]).to(device_output)
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
                force_loss = criterion(out_force, force) * (sub_batch_size / total_batch_size)
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

        val_loss_list.append(val_loss/num_cal)

        if scheduler:
            scheduler.step(val_loss/num_cal)

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
            print(f"        ∆Train: {train_loss_list[-1]-train_loss_list[-2]} - ∆Val: {val_loss_list[-1] - val_loss_list[-2]}")
            # print(f"        ∆Energy: {energy_loss_list[-1]-energy_loss_list[-2]} - ∆Force: {force_loss_list[-1] - force_loss_list[-2]}")

        for k in val_term_losses.keys():
            print(f"        Train {k} loss={train_term_losses[k]/train_term_num_cal[k]:.4f}")
            print(f"        Val   {k} loss={val_term_losses[k]/val_term_num_cal[k]:.4f}")

        if check_early_stopping(val_loss_list[first_early_stopping_epoch:], patience=early_stopping):
            print("Early stopping triggered.")
            break

        history = {"train": train_loss_list, "val": val_loss_list, "energy": energy_loss_list, "force": force_loss_list}

        # Save the model
        # I've attempted to make this compatible with the TorchMD calculators.External class, but I'm not sure how well the keys match - Daniel
        tmp_checkpoint_path = f'{result_directory}/checkpoint-{epoch}.pth'
        save_checkpoint(tmp_checkpoint_path, epoch + 1, model, optimizer, conf, scheduler)

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
        print("        Checkpoint saved.")

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
    parser = argparse.ArgumentParser(description="Train a CGSchNet network")
    parser.add_argument("input", help="Processed data to train on ")
    parser.add_argument("result", default=None, nargs="?", help="Checkpoint directory to continue")
    parser.add_argument("-c", "--config", default="../configs/config.yaml", type=str, help="")
    parser.add_argument("--gpus", default=None, type=str, help="List of GPUs to train on (e.g. \"0,1,2\")")
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

    assert torch.cuda.is_available(), "CUDA is not available, please run on a machine with CUDA or use --gpus cpu"

    args = parser.parse_args()

    directory_path = args.input
    assert os.path.isdir(directory_path), f"Input directory does not exist: {directory_path}"
    result_directory = args.result
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
        train_model(directory_path, result_directory=result_directory, conf_path=conf_path, dry_run=dry_run, weight_decay=weight_decay,
                    learning_rate=learning_rate, gpu_ids=gpu_ids, epochs=epochs, batch_size=batch_size, val_ratio=val_ratio, scheduler=lr_scheduler,
                    atoms_per_call=atoms_per_call, reset_early_stopping=reset_early_stopping, enable_shuffle=enable_shuffle,
                    mini_epoch_size=mini_epoch_size, early_stopping=early_stopping, checkpoint_save=checkpoint_save, subsetpdbs=subsetpdbs, energy_weight=energy_weight,
                    force_weight=force_weight, energy_matching=energy_matching, train_term_def=train_term_def, embedding_filename=embedding_filename,
                    dataset_chunk_size=dataset_chunk_size, use_npfile=use_npfile)
    except Exception as e:
        # Uncaught exceptions cause pytorch to hang for quite a while before exiting
        traceback.print_tb(e.__traceback__)
        print(e)
        sys.exit(1)
