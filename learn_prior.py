#!/usr/bin/env python3

# from torchmd.forcefields.forcefield import ForceField
from module.torchmd.tagged_forcefield import TaggedYamlForcefield
from torchmd.forces import Forces
# from torchmd.systems import System
from moleculekit.molecule import Molecule
from torchmd.parameters import Parameters

import json
import torch
import torch.utils.data
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
from torch.utils.data import Dataset

from collections import defaultdict
from math import radians, degrees
import yaml
import shutil

import glob
import os

### Dataset

class ProteinDataset(Dataset):
    """
    This class provides a Dataset that can pull from multiple trajectories at once and
    arrange the data into batches appropriate for passing to TorchMD.
    """

    def __init__(self, directory, pdb_ids):
        """
        Initialize the dataset.
        
        :param directory: The directory to find the prepared data files in.
        :param pdb_ids: A list of protein subdirectories to load from within "directory".
        """
        self.pdb_ids = pdb_ids
        self.coordinates = []
        self.forces = []
        self.boxes = []

        for pdbid in pdb_ids:
            self.coordinates.append(np.load(f'{directory}/{pdbid}/raw/coordinates.npy'))
            self.forces.append(np.load(f'{directory}/{pdbid}/raw/forces.npy'))
            if os.path.exists(f'{directory}/{pdbid}/raw/box.npy'):
                self.boxes.append(np.load(f'{directory}/{pdbid}/raw/box.npy'))

        if not self.boxes:
            self.boxes = None
        else:
            assert len(self.coordinates) == len(self.boxes), \
            "If any boxes are specified, all trajectories must have boxes."

        for i in range(len(self.coordinates)):
            assert len(self.coordinates[0]) == len(self.coordinates[i]), \
                f"{self.pdb_ids[i]}: All trajectories must have the same number of frames."

            assert len(self.coordinates[i]) == len(self.forces[i]), \
                f"{self.pdb_ids[i]}: The lengths of coordinates and forces do not match."

            if self.boxes:
                assert len(self.coordinates[i]) == len(self.boxes[i]), \
                    f"{self.pdb_ids[i]}: The lengths of coordinates and boxes do not match."


    def has_box(self):
        """
        Returns true if the dataset includes a periodic box
        """
        return self.boxes is not None

    def num_proteins(self):
        """
        Get the number of proteins in the dataset.
        """
        return len(self.coordinates)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.coordinates[0])

    def __getitem__(self, idx):
        coord = [torch.from_numpy(i[idx]).float() for i in self.coordinates]
        force = [torch.from_numpy(i[idx]).float() for i in self.forces]
        boxes = [torch.from_numpy(np.array([])).float() for _ in range(len(self.coordinates))]
        if self.boxes:
            boxes = [torch.from_numpy(i[idx]).float() for i in self.boxes]

        return coord, force, boxes

### LearnableForceField
# Based on torchmd/forcefields/ff_yaml.py

def make_key(at, tagged=False):
    if tagged:
        tagged = all([i.endswith("*") for i in at])
    at = [i.replace("*", "") for i in at]

    if len(at) > 1 and at[0] > at[-1]:
        result = tuple(reversed(at))
    else:
        result = tuple(at)

    if tagged:
        result = tuple([i+"*" for i in result])

    return result

def parse_key(key_str):
    # Turn a string like '(CAA, CAA)' into a sorted tuple
    key_str = key_str[1:-1]
    return make_key(key_str.split(", "), True)

def key_to_str(key):
    return "(" + ", ".join(key) + ")"

class LearnableForceField(nn.Module):
    bond_params: torch.Tensor | None
    angle_params: torch.Tensor | None
    lj_params_sigma: torch.Tensor | None
    lj_params_epsilon: torch.Tensor | None
    def __init__(self, prm, device, train_sigma=False):
        super().__init__()

        with open(prm, "r") as f:
            self.prm = yaml.load(f, Loader=yaml.FullLoader)

        self.bond_params = None
        self.angle_params = None
        self.dihedral_params = None
        self.lj_params_sigma = None
        self.lj_params_epsilon = None

        self.center_angles = False

        def make_indices(keys, parse=True):
            if parse:
                return {parse_key(k): v for v,k in enumerate(sorted(set(keys)))}
            else:
                return {k: v for v,k in enumerate(sorted(set(keys)))}

        # Bonds
        if "bonds" in self.prm:
            self.bond_idx = make_indices(self.prm["bonds"].keys())
            assert len(self.bond_idx) == len(set(self.prm["bonds"].keys()))
            bond_params = np.zeros((len(self.bond_idx), 2), dtype=float)

            for k, v in self.prm["bonds"].items():
                bond_params[self.bond_idx[parse_key(k)]] = (v["k0"], v["req"])

            self.bond_params = nn.Parameter(torch.tensor(bond_params, dtype=torch.float32).to(device))

        # Angles
        if "angles" in self.prm:
            self.angle_idx = make_indices(self.prm["angles"].keys())
            angle_params = np.zeros((len(self.angle_idx), 2), dtype=float)

            first_key = [*self.angle_idx.keys()][0]
            self.center_angles = (first_key[0] == "X" and first_key[1] != "X")

            for k, v in self.prm["angles"].items():
                angle_params[self.angle_idx[parse_key(k)]] = (v["k0"], radians(v["theta0"]))

            self.angle_params = nn.Parameter(torch.tensor(angle_params, dtype=torch.float32).to(device))

        # Dihedrals
        if "dihedrals" in self.prm:
            self.dihedral_idx = make_indices(self.prm["dihedrals"].keys())
            dihedral_params = defaultdict(lambda : np.zeros((len(self.dihedral_idx),2)))
            for k, v in self.prm["dihedrals"].items():
                # print(v)
                for term in v["terms"]:
                    # We don't keep "per" because it's not trainable
                    # [term["phi_k"], radians(term["phase"]), term["per"]]
                    dihedral_params[term["per"]][self.dihedral_idx[parse_key(k)]] = (term["phi_k"], radians(term["phase"]))
            self.dihedral_params = nn.ParameterDict()
            for k, v in dihedral_params.items():
                self.dihedral_params[str(k)] = nn.Parameter(torch.tensor(dihedral_params[k], dtype=torch.float32).to(device))
        
        if "lj" in self.prm:
            self.lj_idx = make_indices(self.prm["lj"].keys(), parse=False)

            lj_params_sigma   = np.zeros((len(self.lj_idx)), dtype=float)
            lj_params_epsilon = np.zeros((len(self.lj_idx)), dtype=float)

            for k, v in self.prm["lj"].items():
                lj_params_sigma[self.lj_idx[k]]   = v["sigma"]
                lj_params_epsilon[self.lj_idx[k]] = v["epsilon"]

            self.lj_params_epsilon = nn.Parameter(torch.tensor(lj_params_epsilon, dtype=torch.float32).to(device))
            if train_sigma:
                self.lj_params_sigma = nn.Parameter(torch.tensor(lj_params_sigma, dtype=torch.float32).to(device))
            else:
                self.lj_params_sigma = torch.tensor(lj_params_sigma, dtype=torch.float32).to(device)

    def clip_parameters(self, epsilon=0.0):
        """Clip parameters to a valid range: k0 >= epsilon, pi >= angle >= epsilon"""
        assert isinstance(self.bond_params, torch.Tensor)
        assert isinstance(self.angle_params, torch.Tensor)
        self.bond_params[:,0] = torch.clamp(self.bond_params[:,0], min=epsilon)
        self.angle_params[:,0] = torch.clamp(self.angle_params[:,0], min=epsilon)
        self.angle_params[:,1] = torch.clamp(self.angle_params[:,1], min=epsilon, max=np.pi)

    def median_normalize(self):
        # This sets all parameters to the median value of their type.
        # The median seemed like a better option than the mean because we know the current
        # forcefields have some crazy outlying values.
        def make_median(ar):
            device = ar.device
            return torch.Tensor(np.median(ar.detach().numpy(), 0)).float().to(device)

        with torch.no_grad():
            if self.bond_params is not None:
                self.bond_params[:] = make_median(self.bond_params)
            if self.angle_params is not None:
                self.angle_params[:] = make_median(self.angle_params)
            if self.dihedral_params is not None:
                for k in self.dihedral_params.keys():
                    self.dihedral_params[k][:] = make_median(self.dihedral_params[k])
            if self.lj_params is not None:
                assert isinstance(self.lj_params, torch.Tensor)
                self.lj_params[:] = make_median(self.lj_params)

    def apply_params(self, mol, parameters):
        # From torchmd/parameters.py
        # This gives us the mapping from the atom indexes back to their type names
        uqatomtypes, indexes = np.unique(mol.atomtype, return_inverse=True)

        # The parameters.bonds,.angles,.dihedrals arrays are identical to the mol.bond,etc. ones
        if self.bond_params is not None:
            bond_indexes = [ self.bond_idx[make_key(i, tagged=True)] for i in uqatomtypes[indexes[mol.bonds]] ]
            assert len(parameters.bond_params) == len(bond_indexes)
            parameters.bond_params = self.bond_params[bond_indexes]

        if self.angle_params is not None:
            if self.center_angles:
                angle_indexes = [ self.angle_idx[make_key(("X", i[1], "X"))] for i in uqatomtypes[indexes[mol.angles]] ]
            else:
                angle_indexes = [ self.angle_idx[make_key(i)] for i in uqatomtypes[indexes[mol.angles]] ]
            assert len(parameters.angle_params) == len(angle_indexes)
            parameters.angle_params = self.angle_params[angle_indexes]

        if self.dihedral_params is not None:
            assert len(parameters.dihedral_params) == len(self.dihedral_params)
            if len(self.dihedral_idx) == 1:
                # Wildcard parameter that uses the same value for all types
                for k in self.dihedral_params.keys():
                    ki = int(k)
                    per = torch.Tensor([ki]).float().to(self.dihedral_params[k].device)
                    param = torch.concatenate([self.dihedral_params[k][0], per])
                    param = param.repeat(len(parameters.dihedral_params[ki-1]["params"]),1)
                    assert parameters.dihedral_params[ki-1]["params"].shape == param.shape
                    parameters.dihedral_params[ki-1]["params"] = param
            else:
                raise NotImplementedError()

        if self.lj_params_epsilon is not None:
            # Lorentz - Berthelot combination rule
            # https://pythoninchemistry.org/sim_and_scat/parameterisation/mixing_rules.html
            lj_idx = [self.lj_idx[make_key([i])[0]] for i in uqatomtypes]
            assert isinstance(self.lj_params_sigma, torch.Tensor)
            sigma = self.lj_params_sigma[lj_idx]
            assert isinstance(self.lj_params_epsilon, torch.Tensor)
            epsilon = self.lj_params_epsilon[lj_idx]
            sigma_table = 0.5 * (sigma + sigma[:, None])
            eps_table = torch.sqrt(epsilon * epsilon[:, None])
            sigma_table_6 = sigma_table**6
            # sigma_table_12 = sigma_table_6 * sigma_table_6
            # A = eps_table * 4 * sigma_table_12
            parameters.A = None
            parameters.B = eps_table * 4 * sigma_table_6

    def forward(self, mol, forces, coords, box):
        replicas = len(coords)
        assert self.bond_params is not None
        device = self.bond_params.device
        forces_out = torch.zeros(replicas, mol.numAtoms, 3, device=device)

        self.apply_params(mol, forces.par)
        pot = forces.compute(coords, box, forces_out)
        
        return pot, forces_out 

    def save(self, filename):
        result = copy.deepcopy(self.prm)
        if self.bond_params is not None:
            bond_result = {}
            for k, v in self.bond_idx.items():
                k0, req = map(float, self.bond_params[v].detach().to("cpu").numpy())
                bond_result[key_to_str(k)] = {"k0":k0, "req":req}
            result["bonds"] = bond_result

        if self.angle_params is not None:
            angle_result = {}
            for k, v in self.angle_idx.items():
                k0, theta0 = map(float, self.angle_params[v].detach().to("cpu").numpy())
                theta0 = degrees(theta0)
                angle_result[key_to_str(k)] = {"k0":k0, "theta0":theta0}
            result["angles"] = angle_result

        # FIXME: Not sure how to correctly handle dihedrals with different numbers of terms,
        #        which will probably require changes to the loader too.
        if self.dihedral_params is not None:
            dihedral_result = {}
            num_terms = len(self.dihedral_params)
            for k, v in self.dihedral_idx.items():
                terms = []
                for i in range(1, num_terms+1):
                    phase, phi_k = map(float, self.dihedral_params[str(i)][v].detach().to("cpu").numpy())
                    phase = degrees(phase)
                    terms.append({"per":i, "phase":phase, "phi_k":phi_k})
                dihedral_result[key_to_str(k)] = {"terms":terms}
            result["dihedrals"] = dihedral_result

        if self.lj_params_epsilon is not None:
            assert self.lj_params_sigma is not None
            lj_result = {}
            sigma = [*map(float, self.lj_params_sigma.detach().to("cpu").numpy())]
            epsilon = [*map(float, self.lj_params_epsilon.detach().to("cpu").numpy())]
            for k, v in self.lj_idx.items():
                lj_result[k] = {"sigma":sigma[v], "epsilon":epsilon[v]}
            result["lj"] = lj_result

        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(result, f)


### Init

class TrainingObject():
    def __init__(self, data_dir, pdb_id, device="cpu", precision=torch.float, prior=None):
        self.pdb_id = pdb_id
        self.mol = Molecule(f"{data_dir}/{pdb_id}/processed/{pdb_id}_processed.psf")
        # if prior:
        #     self.initial_ff = ForceField.create(self.mol, prior)
        # else:
        #     self.initial_ff = ForceField.create(self.mol, f"{data_dir}/{pdb_id}/raw/{pdb_id}_priors.yaml")
        if prior:
            self.initial_ff = TaggedYamlForcefield(self.mol, prior)
        else:
            self.initial_ff = TaggedYamlForcefield(self.mol, f"{data_dir}/{pdb_id}/raw/{pdb_id}_priors.yaml")
        prior_params = json.load(open(f"{data_dir}/{pdb_id}/raw/{pdb_id}_prior_params.json"))
        self.parameters_base = Parameters(self.initial_ff, self.mol, terms=prior_params["forceterms"], precision=precision, device=device)
        self.forces = Forces(copy.deepcopy(self.parameters_base), terms=prior_params["forceterms"], exclusions=prior_params["exclusions"])

class LossHistory:
    def __init__(self):
        self.train = []
        self.val = []

    def append(self, train, val):
        self.train.append(train)
        self.val.append(val)

    def print_epoch(self):
        print(f"Train loss: {self.train[-1]} - Validation loss: {self.val[-1]}")
        if len(self.train) > 1:
            print(f"    ∆Train: {self.train[-1]-self.train[-2]} - ∆Val: {self.val[-1] - self.val[-2]}")

    def save(self, filename):
        np.savez(filename, train_loss=self.train, val_loss=self.val)

    def load(self, filename):
        data = np.load(filename)
        self.train = data["train_loss"].tolist()
        self.val = data["val_loss"].tolist()

def train(data_dir, pdb_ids, checkpoint_dir, n_epochs, batch_size, learning_rate, scheduler_gamma, clip_parameters, initial_loss_calc):
    print("Loading dataset...")
    print(" path =", data_dir)
    if not pdb_ids:
        pdb_ids = [os.path.basename(i) for i in glob.glob(os.path.join(data_dir,"*")) if os.path.exists(os.path.join(i,"raw"))]
    print(" pdbids =", ", ".join(pdb_ids))
    all_data = ProteinDataset(data_dir, pdb_ids)

    # validation ratio
    val_ratio = 0.1
    val_size = int(val_ratio * len(all_data))
    train_size = len(all_data) - val_size

    # Generate the test and validation split with deterministic indices
    generator1 = torch.Generator().manual_seed(12341234)
    val_idx, train_idx = torch.utils.data.random_split(torch.arange(len(all_data)), [val_size, train_size], generator=generator1) #pyright: ignore[reportArgumentType]
    train =  torch.utils.data.Subset(all_data, train_idx) #pyright: ignore[reportArgumentType]
    val =  torch.utils.data.Subset(all_data, val_idx) #pyright: ignore[reportArgumentType]

    # train.py uses the torch_geometric but do we really need that?
    # from torch_geometric.loader import DataLoader
    from torch.utils.data import DataLoader

    train_data = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8,
                            persistent_workers=True, pin_memory=True)
    val_data = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8,
                            persistent_workers=True, pin_memory=True)
    
    device = torch.device("cpu")
    # device = torch.device("cuda")
    # precision = torch.double
    precision = torch.float

    print("Building trainable objects...")
    initial_prior_path = os.path.join(data_dir, "priors.yaml")
    initial_prior_params_path = os.path.join(data_dir, "prior_params.json")
    training_mol_list = [TrainingObject(data_dir, i, device, precision, initial_prior_path) for i in pdb_ids] #pyright: ignore[reportArgumentType]
    learnable_ff = LearnableForceField(initial_prior_path, device)
    # learnable_ff.median_normalize()
    n_atoms = sum([i.mol.numAtoms for i in training_mol_list])

    hist = LossHistory()

    optimizer = optim.Adam(learnable_ff.parameters(), lr=learning_rate)
    scheduler = None
    if scheduler_gamma:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduler_gamma)
    #TODO: Change reduction back to 'mean' after cleaning up training loop?
    criterion = nn.MSELoss(reduction='sum')

    epoch = 0

    ### Training
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "prior_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        # Load checkpoint
        checkpoint_dict = torch.load(checkpoint_path)
        print(f"Loading checkpoint: {checkpoint_path}")
        learnable_ff.load_state_dict(checkpoint_dict["learnable_ff"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint_dict["scheduler"])
        epoch = checkpoint_dict["epoch"]
        hist.load(os.path.join(checkpoint_dir, "history.npz"))
    print(f"Saving to: {checkpoint_dir}")

    shutil.copy(initial_prior_path, os.path.join(checkpoint_dir, "initial_priors.yaml"))
    shutil.copy(initial_prior_params_path, os.path.join(checkpoint_dir, "prior_params.json"))

    def check_early_stopping(_, val_list, patience=2):
        if len(val_list) < patience+1:
            return False
        check_range = np.array(val_list[-(patience+1):])
        if np.all((check_range[1:]-check_range[:-1])>0):
            print(f"Validation loss increased {patience} times, stopping...")
            return True
        return False

    def process_batch(batch, training_mol_list, learnable_ff):
        coords, forces, boxes = batch
        loss = 0.0
        pots = [None for _ in range(len(training_mol_list))]

        for i in range(len(training_mol_list)):
            mo = training_mol_list[i].mol
            replicas = len(coords[i])

            co = torch.Tensor(coords[i]).to(device)
            fo = torch.Tensor(forces[i]).to(device)
            if (torch.numel(boxes[i]) > 0):
                bo = torch.Tensor(boxes[i]).to(device)
            else:
                bo = torch.zeros(replicas, 3, 3).to(device)

            pot, forces_out = learnable_ff.forward(mo, training_mol_list[i].forces, co, bo)
            pots[i] = pot

            # TODO: Should this be scaled by batch size (replicas) or not?
            loss += criterion(forces_out, fo)/(n_atoms*replicas*3)
        return loss, pots

    if epoch == 0 and initial_loss_calc:
        training_loss = 0
        val_loss = 0
        pots = [list() for _ in range(len(training_mol_list))]

        with torch.no_grad():
            for batch in tqdm(train_data, desc="Initial Training Loss"):
                loss, batch_pots = process_batch(batch, training_mol_list, learnable_ff)
                training_loss += loss.detach().item() * batch_size/len(train_data) #pyright: ignore[reportAttributeAccessIssue]
                for i in range(len(pots)):
                    pots[i].extend(batch_pots[i]) #pyright: ignore[reportArgumentType]
            
            for batch in tqdm(val_data, desc="Initial Validation Loss"):
                loss, batch_pots = process_batch(batch, training_mol_list, learnable_ff)
                val_loss += loss.detach().item() * batch_size/len(val_data) #pyright: ignore[reportAttributeAccessIssue]

        hist.append(training_loss, val_loss)
        hist.print_epoch()
        # print("std(pots):", [np.std(i) for i in pots])

    for epoch in range(epoch, n_epochs):
        # print(f"Epoch {epoch}/{n_epochs}")
        training_loss = 0
        val_loss = 0
        pots = [list() for _ in range(len(training_mol_list))]

        for batch in tqdm(train_data, desc=f"Training ({epoch}/{n_epochs})"):
            optimizer.zero_grad()
            loss, batch_pots = process_batch(batch, training_mol_list, learnable_ff)
            for i in range(len(pots)):
                pots[i].extend(batch_pots[i]) #pyright: ignore[reportArgumentType]
            training_loss += loss.detach().item() * batch_size/len(train_data) #pyright: ignore[reportAttributeAccessIssue] 
            loss.backward() #pyright: ignore[reportAttributeAccessIssue]
            optimizer.step()

        if scheduler:
            scheduler.step()
        
        with torch.no_grad():
            # Clip any out of range values
            #FIXME: Currently this must happen inside no_grad()
            if clip_parameters:
                learnable_ff.clip_parameters(0.00001)

            for batch in tqdm(val_data, desc=f"Validation ({epoch}/{n_epochs})"):
                loss, batch_pots = process_batch(batch, training_mol_list, learnable_ff)
                val_loss += loss.detach().item() * batch_size/len(val_data) #pyright: ignore[reportAttributeAccessIssue]

        hist.append(training_loss, val_loss)
        hist.print_epoch()
        # print("std(pots):", [np.std(i) for i in pots])

        if check_early_stopping(hist.train, hist.val, 3):
            print("Early stopping triggered.")
            break

        hist.save(os.path.join(checkpoint_dir, "history.npz"))

        print("Saving checkpoint...")
        checkpoint_dict = {
            "learnable_ff": learnable_ff.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch+1,
        }
        tmp_checkpoint_path = os.path.join(checkpoint_dir, f"prior_checkpoint-{epoch}.pth")
        learnable_ff.save(os.path.join(checkpoint_dir, f"priors-{epoch}.yaml"))
        torch.save(checkpoint_dict, tmp_checkpoint_path)
        if (epoch-1) % 10 == 0 and epoch > 1:
            shutil.copyfile(tmp_checkpoint_path, os.path.join(checkpoint_dir,'prior_checkpoint.pth'))
        else:
            os.replace(tmp_checkpoint_path, os.path.join(checkpoint_dir,'prior_checkpoint.pth'))
        learnable_ff.save(os.path.join(checkpoint_dir, "priors.yaml"))

        # If this is <= to the lowest validation loss seen so far also save it to checkpoint-best.pth
        if hist.val[-1] <= np.min(hist.val):
            shutil.copyfile(os.path.join(checkpoint_dir,'prior_checkpoint.pth'), os.path.join(checkpoint_dir,'prior_checkpoint-best.pth'))
            shutil.copyfile(os.path.join(checkpoint_dir,'priors.yaml'), os.path.join(checkpoint_dir,'priors-best.yaml'))

### End Train

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a TorchMD prior forcefield")
    parser.add_argument("input", help="Preprocessed prior & data to train on")
    parser.add_argument("result", help="Checkpoint save directory")
    parser.add_argument("--pdbids", nargs="*", help="List of specific PDB IDs to process")
    parser.add_argument("--batch", type=int, default=10, help="The batch size to use")
    parser.add_argument("--epochs", type=int, default=25, help="The total number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=None, help="Learning rate scheduler gamma")
    parser.add_argument("--clip-parameters", action='store_true', help="Clip parameters to a valid range after each update")
    parser.add_argument('--initial-loss', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to calculate initial loss before training. Default=True")

    args = parser.parse_args()
    print(args)

    train(data_dir = args.input,
          pdb_ids = args.pdbids,
          checkpoint_dir = args.result,
          n_epochs = args.epochs,
          batch_size = args.batch,
          learning_rate = args.lr,
          scheduler_gamma = args.gamma,
          clip_parameters = args.clip_parameters,
          initial_loss_calc = args.initial_loss)
