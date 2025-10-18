#!/usr/bin/env python3
import os
import numpy as np
import yaml
import json
import h5py
import mdtraj
from module.torchmd_cg_mappings import CACB_MAP
from module import prior
from module import prior_flex
from module import psfwriter
from module.make_deltaforces import DeltaForces
from module.cg_mapping import CGMapping
import argparse
import traceback
import shutil
import multiprocessing as mp
from tqdm import tqdm
import glob
import pickle
import torch
# this can have a small performance hit as it uses the HDD file_system to share data across different processes, but it doesn't lead to "Too many files open" error, which was limiting the max number of parallel processes to 16.
torch.multiprocessing.set_sharing_strategy('file_system') 

# Raz Dec 30 2024: turns out that when preprocessing the 6.5k dataset, the last 20 pdbs are taking forever to process due to being very big proteins. In addition, some were giving hdf5 errors since the batch_generate job didn't properly finish, and this means we don't have all the frames, and they will be removed later on before training. So here's the best workflow I found to solve it:
# 1) first run just step 1 on all proteins, and set FILTER_NOT_PROCESSED_STEP_ONE = False. if any pdbs are taking forever (generally the last 20), just kill the job
# 2) re-run the pre-processing for a second time with FILTER_NOT_PROCESSED_STEP_ONE = True 
FILTER_NOT_PROCESSED_STEP_ONE = False

# Controls whether to re-generate the priors in step 2 for each of the terms, terms in the list will be loaded from the cache instead of refit
#USE_CACHED_FITS = ['dihedrals', 'angles', 'bonds', 'lj']
USE_CACHED_FITS = []

DEVICE_STEP_3 = 'cpu'
#DEVICE_STEP_3 = 'cuda'

DO_STEP_1 = True # whether to do step 1. if you got errors in steps 2-3 and want to resume, set this to False
REGEN_CACHE_FILES = True # whether to re-generate cache files


def process_init(counter):
    """This function sets the worker names such that we can use them to position the tqdm bars"""
    with counter.get_lock():
        idx = int(counter.value)
        counter.value += 1
    mp.current_process().name = f"PreprocessWorker-{idx}"


class CGMappingDef_CA:
    def __init__(self):
        residues = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "HYP", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
        # For legacy reasons we have a couple extra ambiguous residues (ASX & GLX) in the embedding map but we do not accept these for parsing
        embedding_residues = ["ALA", "ARG", "ASN", "ASP", "ASX", "CYS", "GLU", "GLN", "GLX", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
        self.bead_embeddings = {name: [index + 1] for index, name in enumerate(sorted(embedding_residues))}

        # bead_atom_selection: A list of lists, where each inner list is the names of the atoms that will be combined to form the bead
        self.bead_atom_selection = {k: [["CA"]] for k in residues}
        # The type names of beads (will become the atom type/element in the cg topology)
        self.bead_types = {
            "ALA": ["CAA"],
            "ARG": ["CAR"],
            "ASN": ["CAN"],
            "ASP": ["CAD"],
            "CYS": ["CAC"],
            "GLN": ["CAQ"],
            "GLU": ["CAE"],
            "GLY": ["CAG"],
            "HIS": ["CAH"],
            "HSD": ["CAH"],
            "ILE": ["CAI"],
            "LEU": ["CAL"],
            "LYS": ["CAK"],
            "MET": ["CAM"],
            "PHE": ["CAF"],
            "PRO": ["CAP"],
            "SER": ["CAS"],
            "THR": ["CAT"],
            "TRP": ["CAW"],
            "TYR": ["CAY"],
            "VAL": ["CAV"],
        }
        # The "atom name" assigned to the beads
        self.bead_atom_names = {k: ["CA"] for k in residues}
        self.bead_masses = {k: [12.01] for k in residues}
        self.bead_backbone_idx = {k: 0 for k in residues}

class CGMappingDef_CACB:
    def __init__(self):
        residues = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "HYP", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]

        # bead_atom_selection: A list of lists, where each inner list is the names of the atoms that will be combined to form the bead
        self.bead_atom_selection = {k: [["CA"], ["CB"]] for k in residues}
        self.bead_atom_selection["GLY"] = [["CA"]]
        # The type names of beads (will become the atom type/element in the cg topology)
        self.bead_types = {
            "ALA": ["CA", "CBA"],
            "ARG": ["CA", "CBR"],
            "ASN": ["CA", "CBN"],
            "ASP": ["CA", "CBD"],
            "CYS": ["CA", "CBC"],
            "GLN": ["CA", "CBQ"],
            "GLU": ["CA", "CBE"],
            "GLY": ["CAG"],
            "HIS": ["CA", "CBH"],
            "HSD": ["CA", "CBH"],
            "ILE": ["CA", "CBI"],
            "LEU": ["CA", "CBL"],
            "LYS": ["CA", "CBK"],
            "MET": ["CA", "CBM"],
            "PHE": ["CA", "CBF"],
            "PRO": ["CA", "CBP"],
            "SER": ["CA", "CBS"],
            "THR": ["CA", "CBT"],
            "TRP": ["CA", "CBW"],
            "TYR": ["CA", "CBY"],
            "VAL": ["CA", "CBV"],
        }

        embedding_map = {k:i for i,k in enumerate(sorted(set.union(*[set(i) for i in self.bead_types.values()])))}
        self.bead_embeddings = {k:[embedding_map[i] for i in v] for k, v in self.bead_types.items()}

        # The "atom name" assigned to the beads
        self.bead_atom_names = {k: ["CA", "CB"] for k in residues}
        self.bead_atom_names["GLY"] = ["CA"]
        self.bead_masses = {k: [12.01]*len(v) for k,v in self.bead_types.items()}
        self.bead_backbone_idx = {k: 0 for k in residues}

class PriorBuilder:
    def __init__(self):
        self.prior_params = dict()
        self.priors = None
        self.terms = dict()
        self.atom_types = set()
        self.fit_constraints = True
        self.tag_beta_turns = False
        self.min_cnt = 0

    def select_atoms(self, topology):
        """Returns tha atom index to be saved for this prior"""
        raise NotImplementedError()

    def map_embeddings(self, selected_atoms, trajectory):
        """Generates the embeddings array for the selected atoms"""
        raise NotImplementedError()

    def write_psf(self, pdb_file, psf_file):
        """Write the .psf file describing the course grain geometry"""
        raise NotImplementedError()

    def add_molecule(self, mol, traj, cache_dir):
        fit_ok_path = os.path.join(cache_dir, "fit_ok.txt")

        if cache_dir and os.path.exists(fit_ok_path):
            os.unlink(fit_ok_path)

        for term in self.terms.values():
            term.add_molecule(mol, traj, cache_dir)
        self.atom_types = self.atom_types.union(mol.atomtype)

        if cache_dir:
            np.save(os.path.join(cache_dir, "atomtype.npy"), mol.atomtype)
            with open(fit_ok_path, "wt", encoding="utf-8") as f:
                f.write("ok")

    def load_molecule_cache(self, cache_dir):
        assert os.path.exists(os.path.join(cache_dir, "fit_ok.txt"))
        atomtype = np.load(os.path.join(cache_dir, "atomtype.npy"), allow_pickle=True)
        self.atom_types = self.atom_types.union(atomtype)

        for term in self.terms.values():
            term.load_molecule_cache(cache_dir)

    def enable_fit_constraints(self, use_constraints):
        self.fit_constraints = use_constraints
        self.prior_params["fit_constraints"] = self.fit_constraints

    def enable_bond_tags(self, use_tags):
        self.tag_beta_turns = use_tags
        self.prior_params["tag_beta_turns"] = self.tag_beta_turns

    def set_min_cnt(self, min_cnt):
        assert min_cnt >= 0
        self.min_cnt = min_cnt
        self.prior_params["min_cnt"] = self.min_cnt

    def fit(self, temperature, plot_dir=None):
        self.init_prior_dict()
        assert self.priors is not None
        for key, term in self.terms.items():
            if os.path.exists(f"{plot_dir}/prior_{key}.pkl") and (key in USE_CACHED_FITS):
                print(f"Used cached fit for {key}...")
                with open(f"{plot_dir}/prior_{key}.pkl", "rb") as f:
                    self.priors[key] = pickle.load(f)
            else:
                print(f"Fitting {key}...")
                self.priors[key] = term.get_param(temperature, plot_dir, self.fit_constraints, self.min_cnt)
                # pickle the prior for this term
                with open(f"{plot_dir}/prior_{key}.pkl", "wb") as f:
                    pickle.dump(self.priors[key], f)

    def init_prior_dict(self):
        # Define the force field dict
        priors = {}
        priors['atomtypes'] = sorted(self.atom_types)
        priors['bonds'] = {}
        priors['angles'] = {}
        priors['dihedrals'] = {}
        priors['lj'] = {}
        # For mass and charge assume everything is a carbon atom
        priors['electrostatics'] = {at: {'charge': 0.0} for at in priors['atomtypes']}
        # The mass of carbon used here is the from OpenMM/AMBER-14 value
        priors['masses'] = {at: 12.01 for at in priors['atomtypes']}
        self.priors = priors

    def save_prior(self, output_path, pdbid):
        prefix = ""
        if pdbid:
            prefix = f"{pdbid}_"
        with open(os.path.join(output_path, f"{prefix}priors.yaml"), "w") as f:
            yaml.dump(self.priors, f)
        with open(os.path.join(output_path, f"{prefix}prior_params.json"),"w") as f:
            json.dump(self.prior_params, f)

    def make_mol(self, cg_map):
        bonds = "bonds" in self.terms
        angles = "angles" in self.terms
        dihedrals = "dihedrals" in self.terms
        return cg_map.to_mol(bonds = bonds, angles = angles, dihedrals = dihedrals)

class Prior_CA(PriorBuilder):
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA",
            "exclusions" : ['bonds'],
            "forceterms" : ["bonds"],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()

    def build_mapping(self, topology):
        return CGMapping(topology, CGMappingDef_CA())

    def select_atoms(self, topology):
        #TODO: Remove this function (replaced by build_mapping)
        return topology.select('name CA and protein')

    def map_embeddings(self, selected_atoms, topology): #pyright: ignore[reportIncompatibleMethodOverride]
        #TODO: Remove this function (replaced by build_mapping)
        standardResidues = {"ALA", "ARG", "ASN", "ASP", "ASX", "CYS", "GLU", "GLN", "GLX", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
        amino_acid_mapping = {name: index + 1 for index, name in enumerate(sorted(standardResidues))}

        result = []
        for a_idx in selected_atoms:
            r_name = topology.atom(a_idx).residue.name
            result.append(amino_acid_mapping[r_name])
        return np.array(result, dtype=int)

    def write_psf(self, pdb_file, psf_file):
        #TODO: Remove this function (replaced by build_mapping)
        bonds = "bonds" in self.terms
        angles = "angles" in self.terms
        dihedrals = "dihedrals" in self.terms
        return psfwriter.pdb2psf_CA(pdb_file, psf_file, bonds = bonds, angles = angles, dihedrals = dihedrals,
                                    tag_beta_turns = self.tag_beta_turns)

class Prior_CACB(PriorBuilder):
    """Implements the torchmd-cg CACB prior"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CACB",
            "exclusions" : ['bonds'],
            "forceterms" : ["bonds"],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()

    def build_mapping(self, topology):
        return CGMapping(topology, CGMappingDef_CACB())

    def select_atoms(self, topology):
        #TODO: Remove this function (replaced by build_mapping)
        return topology.select('(name CA or name CB) and protein')

    def map_embeddings(self, selected_atoms, topology):#pyright: ignore[reportIncompatibleMethodOverride]
        #TODO: Remove this function (replaced by build_mapping)

        # Make a map from embedding name to embedding name number
        # e.g. {"CAA":0, "CAC":1, ...}
        embedding_map = CACB_MAP
        embedding_nums = dict([(k, i) for i, k in enumerate(sorted(set(embedding_map.values())))])

        result = []
        for a_idx in selected_atoms:
            r_name = topology.atom(a_idx).residue.name
            a_name = topology.atom(a_idx).name
            emb_name = embedding_map[(r_name, a_name)]
            result.append(embedding_nums[emb_name])
        return np.array(result, dtype=int)

    def write_psf(self, pdb_file, psf_file):
        #TODO: Remove this function (replaced by build_mapping)
        bonds = "bonds" in self.terms
        angles = "angles" in self.terms
        dihedrals = "dihedrals" in self.terms
        return psfwriter.pdb2psf_CACB(pdb_file, psf_file, bonds = bonds, angles = angles, dihedrals = dihedrals)

class Prior_CACB_lj(Prior_CACB):
    """torchmd-cg CACB prior with Bonded & RepulsionCG terms"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CACB_lj",
            "exclusions" : ['bonds'],
            "forceterms" : ['bonds', 'repulsioncg'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])

class Prior_CACB_lj_angle_dihedral(Prior_CACB):
    """torchmd-cg CACB prior with Bonded, Angle, Dihedral & RepulsionCG terms"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CACB_lj_angle_dihedral",
            "exclusions" : ['bonds', 'angles', 'dihedrals'],
            "forceterms" : ['bonds', 'angles', 'dihedrals', 'repulsioncg'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior.ParamAngleCalculator()
        self.terms["dihedrals"] = prior.ParamDihedralCalculator()

class Prior_CA_lj(Prior_CA):
    """CA prior with Bonded & RepulsionCG terms"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj",
            "exclusions" : ['bonds'],
            "forceterms" : ['bonds', 'repulsioncg'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])

class Prior_CA_lj_angle(Prior_CA):
    """CA prior with Bonded, Angle, and RepulsionCG terms"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_angle",
            "exclusions" : ['bonds', 'angles'],
            "forceterms" : ['bonds', 'angles', 'repulsioncg'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms['angles'] = prior.ParamAngleCalculator()

class Prior_CA_lj_angle_dihedral(Prior_CA):
    """torchmd-cg CA prior with Bonded, Angle, Dihedral & RepulsionCG terms"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_angle_dihedral",
            "exclusions" : ['bonds', 'angles', 'dihedrals'],
            "forceterms" : ['bonds', 'angles', 'dihedrals', 'repulsioncg'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior.ParamAngleCalculator()
        self.terms["dihedrals"] = prior.ParamDihedralCalculator()

class Prior_CA_lj_angle_dihedralX(Prior_CA):
    """torchmd-cg CA prior with Bonded, Angle, DihedralX & RepulsionCG terms"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_angle_dihedralX",
            "exclusions" : ['bonds', 'angles', 'dihedrals'],
            "forceterms" : ['bonds', 'angles', 'dihedrals', 'repulsioncg'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior.ParamAngleCalculator()
        self.terms["dihedrals"] = prior.ParamDihedralCalculator(unified=True)

class Prior_CA_lj_angleXCX_dihedralX(Prior_CA):
    """torchmd-cg CA prior with Bonded, Angle, DihedralX & RepulsionCG terms"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_angleXCX_dihedralX",
            "exclusions" : ['bonds', 'angles', 'dihedrals'],
            "forceterms" : ['bonds', 'angles', 'dihedrals', 'repulsioncg'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior.ParamAngleCalculator(center=True)
        self.terms["dihedrals"] = prior.ParamDihedralCalculator(unified=True)

class Prior_CA_lj_angleXCX_dihedralX_flex(Prior_CA):
    """torchmd-cg CA prior with highly flexible Bonded, Angle, DihedralX & RepulsionCG terms that fit the data.

    """
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_angleXCX_dihedralX_flex",
            "exclusions" : ['bonds', 'angles', 'dihedrals'],
            "forceterms_nn" : ['bonds', 'angles', 'dihedrals'],
            "forceterms_classical": ['repulsioncg'], # changed from lj, would need to re-generated the dataset (Jan 10 2025). repulsioncg is using just the repulsion term from lj. it uses the same parameters as lj, so need to make sure the right function is evaluated.
            "external" : True
        })
        self.prior_params['forceterms'] = self.prior_params['forceterms_classical'] + self.prior_params['forceterms_nn']

        self.terms["bonds"] = prior_flex.ParamBondedFlexCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior_flex.ParamAngleFlexCalculator(center=True)
        self.terms["dihedrals"] = prior_flex.ParamDihedralFlexCalculator(unified=True)

    # have to override this method since we're saving neural nets as priors
    def save_prior(self, output_path, pdbid):
        prefix = ""
        # if pdbid:
        #     prefix = f"{pdbid}_"
        with open(os.path.join(output_path, f"{prefix}prior_params.json"),"w") as f:
            json.dump(self.prior_params, f)

        # print('self.priors', self.priors.keys())
        # remove the dihedrals and bonds from the priors
        priorsTruncated = self.priors.copy()
        priorsTruncated.pop('dihedrals')
        priorsTruncated.pop('bonds')
        priorsTruncated.pop('angles')
        # print('priorsTruncated', priorsTruncated.keys())

        # save the classical priors using yaml. this is requires because the classical priors are built from the yaml files
        with open(os.path.join(output_path, f"{prefix}priors.yaml"), "w") as f:
            yaml.dump(priorsTruncated, f)

        self.priors['terms'] = self.terms
        self.priors['prior_params'] = self.prior_params

        # also save with pickle
        with open(os.path.join(output_path, f"{prefix}priors.pkl"), "wb") as f:
            pickle.dump(self.priors, f)

    
    def load_prior_nnets(self, output_path):
        # load the prior with pickle
        with open(os.path.join(output_path, "priors.pkl"), "rb") as f:
            self.priors = pickle.load(f)

        # return self.priors
        
        # with open(os.path.join(output_path, f"{prefix}priors.pkl"), "wb") as f:
        #     pickle.dump(self.priors, f)



class Prior_CA_lj_angleXCX_dihedralX_V1(Prior_CA):
    """torchmd-cg CA prior with Bonded, Angle, DihedralX & RepulsionCG terms"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_angleXCX_dihedralX_V1",
            "exclusions" : ['bonds', 'angles', '1-4'],
            "forceterms" : ['Bonds', 'angles', 'dihedrals', 'RepulsionCG'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior.ParamAngleCalculator(center=True)
        self.terms["dihedrals"] = prior.ParamDihedralCalculator(unified=True)

class Prior_CA_lj_bondNull_angleXCX_dihedralX(Prior_CA):
    """torchmd-cg CA prior with Angle, DihedralX & RepulsionCG terms (+ bond exclusions)"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_bondNull_angleXCX_dihedralX",
            "exclusions" : ['bonds', 'angles', '1-4'],
            "forceterms" : ['Bonds', 'angles', 'dihedrals', 'RepulsionCG'],
        })
        self.terms["bonds"] = prior.NullParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior.ParamAngleCalculator(center=True)
        self.terms["dihedrals"] = prior.ParamDihedralCalculator(unified=True)

class Prior_CA_lj_bondNull_angleNull_dihedralX(Prior_CA):
    """torchmd-cg CA prior with DihedralX & RepulsionCG terms (+ bond & angle exclusions)"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_bondNull_angleNull_dihedralX",
            "exclusions" : ['bonds', 'angles', '1-4'],
            "forceterms" : ['Bonds', 'angles', 'dihedrals', 'RepulsionCG'],
        })
        self.terms["bonds"] = prior.NullParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior.NullParamAngleCalculator()
        self.terms["dihedrals"] = prior.ParamDihedralCalculator(unified=True)

class Prior_CA_lj_bondNull_angleNull_dihedralNull(Prior_CA):
    """torchmd-cg CA prior with RepulsionCG terms (+ bond, angle, & dihedral exclusions)"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_bondNull_angleNull_dihedralNull",
            "exclusions" : ['bonds', 'angles', '1-4'],
            "forceterms" : ['Bonds', 'angles', 'dihedrals', 'RepulsionCG'],
        })
        self.terms["bonds"] = prior.NullParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior.NullParamAngleCalculator()
        self.terms["dihedrals"] = prior.NullParamDihedralCalculator()

class Prior_CA_lj_angleNull_dihedralX(Prior_CA):
    """torchmd-cg CA prior with Bonded, DihedralX & RepulsionCG terms (+ angle exclusions)"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_angleNull_dihedralX",
            "exclusions" : ['bonds', 'angles', '1-4'],
            "forceterms" : ['Bonds', 'angles', 'dihedrals', 'RepulsionCG'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior.NullParamAngleCalculator()
        self.terms["dihedrals"] = prior.ParamDihedralCalculator(unified=True)

class Prior_CA_lj_angleNull_dihedralNull(Prior_CA):
    """torchmd-cg CA prior with Bonded & RepulsionCG terms (+ angle & dihedral exclusions)"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_angleNull_dihedralNull",
            "exclusions" : ['bonds', 'angles', '1-4'],
            "forceterms" : ['Bonds', 'angles', 'dihedrals', 'RepulsionCG'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["angles"] = prior.NullParamAngleCalculator()
        self.terms["dihedrals"] = prior.NullParamDihedralCalculator()

class Prior_CA_Majewski2022_v0(Prior_CA):
    """torchmd-cg CA prior based on the parameters used in (Majewski 2022)
    Note this version (v0) has different lj exclusions than the one used in the paper.
    """
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_Majewski2022_v0",
            "exclusions" : ['bonds', 'dihedrals'],
            "forceterms" : ['bonds', 'dihedrals', 'repulsioncg'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        self.terms["dihedrals"] = prior.ParamDihedralCalculator(unified=True, scale=0.5)

class Prior_CA_Majewski2022_v1(Prior_CA):
    """torchmd-cg CA prior based on the parameters used in (Majewski 2022)"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_Majewski2022_v1",
            "exclusions" : ['bonds'],
            "forceterms" : ['bonds', 'dihedrals', 'repulsioncg'],
        })
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6], exclusion_terms={"bonds"})
        self.terms["dihedrals"] = prior.ParamDihedralCalculator(unified=True, scale=0.5)

class Prior_CA_null(Prior_CA):
    """CA prior with no terms"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_null",
            "exclusions" : [],
            "forceterms" : [],
        })
        self.terms = {}

class Prior_CA_lj_only(Prior_CA):
    """CA prior with just a RepulsionCG term"""
    def __init__(self):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_only",
            "exclusions" : [],
            "forceterms" : ['RepulsionCG'],
        })
        self.terms = {}
        self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])

def slice_to_str(s):
    result = [s.start, s.stop, s.step]
    result = [str(i) if i is not None else '' for i in result]
    return ":".join(result)

def get_prior_params_path(prior_path):
    dir_path, file_name = os.path.split(prior_path)
    file_name = file_name.replace("priors.yaml", "prior_params.json")
    return os.path.join(dir_path, file_name)

def load_h5_traj_slice(path, slice):
    """Load a slice from a h5 trajectory without reading the entire file into memory"""
    base_traj = mdtraj.load_frame(path, 0)
    with h5py.File(path) as f:
        t_xyz = f["coordinates"][slice][:] #pyright: ignore[reportIndexIssue]
        t_time = f["time"][slice][:] #pyright: ignore[reportIndexIssue]

        t_unitcell_lengths = None
        t_unitcell_angles = None
        if "cell_lengths" in f.keys():
            t_unitcell_lengths = f["cell_lengths"][slice][:] #pyright: ignore[reportIndexIssue]
            t_unitcell_angles = f["cell_angles"][slice][:] #pyright: ignore[reportIndexIssue]

    result = mdtraj.Trajectory(t_xyz, base_traj.topology, time=t_time, unitcell_lengths=t_unitcell_lengths, unitcell_angles=t_unitcell_angles)
    return result

class Preprocessor:
    def __init__(self, dataset_conf, input_path_map, save_path, prior_builder, prior_file, prior_name, frame_slice, temp, optimize_forces, box, prior_plots, resume_preprocess, num_cores, jobid=None, totalNrJobs=None):
        self.dataset_conf = dataset_conf
        self.save_path = save_path
        self.prior_builder = prior_builder
        self.prior_file = prior_file
        self.frame_slice = frame_slice
        self.temp = temp
        self.jobid = jobid
        self.totalNrJobs = totalNrJobs

        self.pdbid_list = input_path_map

        if FILTER_NOT_PROCESSED_STEP_ONE:
            pdbs_processed_step1 = [f.split('/')[-3] for f in glob.glob(os.path.join(save_path, "*/fit/fit_ok.txt"))]
            # remove keys that are not in pdbs_processed_step1
            self.pdbid_list = {k: v for k, v in self.pdbid_list.items() if k in pdbs_processed_step1}
            print('%d pdbs left after removing pdbs not processed in step 1' % len(self.pdbid_list))

        # if os.path.exists(os.path.join(save_path, 'pdb_list.pkl')):
        #     with open(os.path.join(save_path, 'pdb_list.pkl'), 'rb') as f:
        #         self.pdbid_list = pickle.load(f)
        # else:
        #     self.pdbid_list = self.get_pdbid_list()
            
        #     if FILTER_NOT_PROCESSED_STEP_ONE:
        #         pdbs_processed_step1 = [f.split('/')[-3] for f in glob.glob(os.path.join(save_path, "*/fit/fit_ok.txt"))]
        #         # remove keys that are not in pdbs_processed_step1
        #         self.pdbid_list = {k: v for k, v in self.pdbid_list.items() if k in pdbs_processed_step1}
        #         print('%d pdbs left after removing pdbs not processed in step 1' % len(self.pdbid_list))

        #     # pickle pdb_list
        #     os.makedirs(save_path, exist_ok=True)
        #     with open(os.path.join(save_path, 'pdb_list.pkl'), 'wb') as f:
        #         pickle.dump(self.pdbid_list, f)


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
        # print("PDB ID list:", self.pdbid_list)

    def step1_threading(self, pdbid):
        try:
            cache_dir = os.path.join(self.save_path, pdbid, "fit")
            if not(self.resume_preprocess and os.path.exists(os.path.join(cache_dir, "fit_ok.txt"))):
                # This assumes we've named the processes during initialization
                bar_pos = int(mp.current_process().name.split("-")[1]) + 1
                self.process_step1(pdbid, bar_pos)
                return []
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(f"{pdbid}:", e)
            raise

    def step3_threading(self, pdbid):
        try:
            # This assumes we've named the processes during initialization
            bar_pos = int(mp.current_process().name.split("-")[1]) + 1
            self.process_step3(pdbid, bar_pos)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(f"{pdbid}:", e)
            raise

    def preprocess(self):
        os.makedirs(os.path.join(self.save_path, "result"), exist_ok=True)

        info_dict = {
            "input_paths":     [i["path"] for i in self.dataset_conf],
            "frame_slice":     slice_to_str(self.frame_slice),
            "pdbids":          list(self.pdbid_list.keys()),
            "optimize_forces": self.optimize_forces,
            "box":             self.box,
            "prior_name":      prior_name
        }

        # If resuming, validate that no paramters that would invalidate the fit cache object have changed
        # FIXME: This should also check "--tag-beta-turns"
        if self.resume_preprocess:
            if os.path.exists(os.path.join(self.save_path, "result/info.json")):
                with open(os.path.join(self.save_path, "result/info.json"), "rt", encoding="utf-8") as f:
                    prevous_info = json.load(f)
                    for k in ["box", "frame_slice", "optimize_forces", "prior_name"]:
                        assert info_dict[k] == prevous_info[k], \
                            f"Can't resume with different parameters: {k}: {info_dict[k]} != {prevous_info[k]}"

        with open(os.path.join(self.save_path, "result/info.json"), "wt", encoding="utf-8") as f:
            json.dump(info_dict, f)

        pdbids = self.pdbid_list

        # Ensure all jobs have the same tqdm lock : https://github.com/tqdm/tqdm/issues/982
        tqdm.get_lock()

        # TODO: Print exceptions in the main thread for legibility
        # TODO: Abstract the loop logic instead of repeating it twice

        # Truncate any existing ok_list.txt
        with open(os.path.join(self.save_path, "result/ok_list.txt"), "wt", encoding="utf-8") as ok_list:
            pass



        # Run step 1 in parallel, saving the results to the cache

        if DO_STEP_1:
            errorList = {}
            thread_counter = mp.Value('i', 0, lock=True)
            with tqdm(total=len(pdbids), desc="Processing Step 1", dynamic_ncols=True) as pbar:
                with mp.Pool(self.num_cores, initializer=process_init, initargs=(thread_counter,)) as pool:
                    pending_results = {}
                    
                    # Submit tasks and map results to pdbids
                    for pdbid in pdbids:
                        result = pool.apply_async(self.step1_threading, args=(pdbid,))
                        pending_results[result] = pdbid

                    while pending_results:
                        # Check completed tasks
                        for result in list(pending_results.keys()):  # Iterate over a copy to allow removal
                            if result.ready():
                                try:
                                    result.get()  # Retrieve result or raise exception
                                except Exception as e:
                                    pdbid = pending_results[result]  # Get the corresponding pdbid
                                    errorList[pdbid] = str(e)
                                finally:
                                    del pending_results[result]  # Remove completed task

                        # Update the progress bar
                        pbar.n = len(pdbids) - len(pending_results)
                        pbar.refresh()

                        # Wait for 1 second or for the last job to finish
                        if pending_results:
                            next(iter(pending_results)).wait(1)

            if len(errorList):
                print('errorList', errorList)
                print('errorList keys', errorList.keys())

        if not self.prior_file:
            # pickle prior builder
            if not os.path.exists(self.save_path + '/prior_builder.pkl') or REGEN_CACHE_FILES:
                # Merge cache files back into prior builder
                for pdbid in tqdm(pdbids, desc="Merging cache files together"):
                    cache_dir = os.path.join(self.save_path, pdbid, "fit")
                    self.prior_builder.load_molecule_cache(cache_dir)

                with open(self.save_path + '/prior_builder.pkl', 'wb') as f:
                    pickle.dump(self.prior_builder, f)
            else:
                print("Using cached prior_builder object... ")

                with open(self.save_path + '/prior_builder.pkl', 'rb') as f:
                    self.prior_builder = pickle.load(f)

            self.process_step2()

            self.prior_builder.save_prior(self.save_path, None)
        else:
            prior_params_path = get_prior_params_path(prior_file)
            shutil.copy(self.prior_file, os.path.join(self.save_path, "priors.yaml"))
            shutil.copy(prior_params_path, os.path.join(self.save_path, "prior_params.json"))
        
        if self.totalNrJobs:
            pdblist = list(pdbids.keys())
            pdbidsPerJob = len(pdblist) // self.totalNrJobs + 1
            jobid = self.jobid
            assert jobid is not None
            if jobid < self.totalNrJobs - 1:
                pdbids_c = [pdblist[i] for i in range(jobid * pdbidsPerJob, (jobid + 1) * pdbidsPerJob)]
            else:
                pdbids_c = [pdblist[i] for i in range(jobid * pdbidsPerJob, len(pdblist))]

            # filter pdbids for this job only
            pdbids = {k: v for k, v in pdbids.items() if k in pdbids_c}
        
        print(f"Step 3: Processing {len(pdbids)} pdbids")

        # Run step 3 in parallel
        thread_counter = mp.Value('i', 0, lock=True)
        with tqdm(total=len(pdbids), desc="Processing Step 3", dynamic_ncols=True) as pbar:
            with mp.Pool(self.num_cores, initializer=process_init, initargs=(thread_counter,)) as pool:
                pending_results = []

                for pdbid in pdbids:
                    pending_results += [pool.apply_async(self.step3_threading, args=(pdbid,))]

                while pending_results:
                    # Check for exceptions
                    [i.get() for i in pending_results if i.ready()]
                    # Remove finished jobs from the list
                    pending_results = [i for i in pending_results if not i.ready()]
                    pbar.n = len(pdbids) - len(pending_results)
                    pbar.refresh()
                    # Wait for 1 second or for the last job to finish
                    if pending_results:
                        pending_results[0].wait(1)

        # alternatively, cd to the preprocessed_data directory and run this cmd:
        #  ls */raw/deltaforces.npy | awk '{print substr($1, 1, 4)}' > result/ok_list.txt
        with open(os.path.join(self.save_path, "result/ok_list.txt"), "wt", encoding="utf-8") as ok_list:
            ok_list.write("\n".join(pdbids))

        print("Done!")

    def save_data(self, output_path, trajectory, embeddings, forces, pdbid):
        # print(f"  {pdbid} (coordinates, forces): {trajectory.xyz.shape}, {forces.shape}")
        np.save(f"{output_path}/raw/embeddings.npy", embeddings)
        np.save(f"{output_path}/raw/forces.npy", forces)
        np.save(f"{output_path}/raw/coordinates.npy", trajectory.xyz)
        box_path = f"{output_path}/raw/box.npy"
        if self.box:
            np.save(box_path, trajectory.unitcell_vectors)
        elif os.path.exists(box_path):
            os.unlink(box_path)

    def process_step1(self, pdbid, bar_position=0):
        """Generate the course grained data and topology for the protein, then add it to the prior builder"""

        with tqdm(total=7, position=bar_position, desc=f"{pdbid}: File path setup", dynamic_ncols=True, leave=False) as pbar:
            def progress_bar_step(msg):
                pbar.update(1)
                pbar.set_description_str(f"{pdbid}: {msg}")

            # Set up paths and create directories
            output_path = os.path.join(self.save_path, pdbid)

            # TODO: Get rit of the of subdirectories?
            os.makedirs(f"{output_path}/raw", exist_ok=True)
            os.makedirs(f"{output_path}/processed", exist_ok=True)

            # Find which path this ID belongs to
            input_file_path = self.pdbid_list[pdbid]

            progress_bar_step("Loading trajectory")
            AAtraj = load_h5_traj_slice(input_file_path, self.frame_slice)
            assert AAtraj.xyz is not None # for pyright
            AAtraj.xyz *= 10  # convert to angstroms

            progress_bar_step("Building CG mapping")
            cg_map = self.prior_builder.build_mapping(AAtraj.topology)
            mol = self.prior_builder.make_mol(cg_map)
            topology = cg_map.to_mol(bonds=True, angles=True, dihedrals=True)
            mol.write(f'{output_path}/processed/{pdbid}_processed.psf')
            topology.write(f'{output_path}/processed/topology.psf')  # Save the topology for the CG mapping, this is optional but useful for debugging

            # Get the forces
            progress_bar_step("Mapping CG forces")
            with h5py.File(input_file_path, "r") as f:
                forces = f["forces"][self.frame_slice, :, :] #pyright: ignore[reportIndexIssue]

                if self.optimize_forces:
                    forces = cg_map.cg_optimal_forces(AAtraj, forces)
                else:
                    forces = cg_map.cg_forces(forces)

                assert len(forces) == len(AAtraj)
                # Convert from kilojoules/mole/nanometer to kilocalories/mole/angstrom
                forces = forces*0.02390057361376673

            progress_bar_step("Mapping CG coordinates")
            xyz = cg_map.cg_positions(AAtraj.xyz)
            cg_traj = mdtraj.Trajectory(xyz, topology=cg_map.to_mdtraj())
            if self.box and AAtraj.unitcell_lengths is not None:
                cg_traj.unitcell_lengths = AAtraj.unitcell_lengths * 10
                cg_traj.unitcell_angles  = AAtraj.unitcell_angles
            else:
                cg_traj.unitcell_lengths = None
                cg_traj.unitcell_angles  = None

            # Save the data
            progress_bar_step("Saving Data")
            self.save_data(output_path, cg_traj, cg_map.embeddings, forces, pdbid)

            # Note: moveaxis creates a view, the original trajectory.xyz is unmodified
            assert cg_traj.xyz is not None
            mol.coords = np.moveaxis(cg_traj.xyz, 0, -1)

            progress_bar_step("Generating prior fit data")
            if not self.prior_file:
                cache_dir = os.path.join(output_path, "fit")
                os.makedirs(cache_dir, exist_ok=True)
                self.prior_builder.add_molecule(mol, cg_traj, cache_dir)

    def process_step2(self):
        """Fit the prior forcefield based on accumulated data"""
        if self.prior_plots:
            plot_dir = os.path.join(self.save_path, "prior_fit_plots")
            os.makedirs(plot_dir, exist_ok=True)
        else:
            plot_dir = None

        self.prior_builder.fit(self.temp, plot_dir=plot_dir)

    def process_step3(self, pdbid, bar_position=0):
        """Save prior focefield and generate delta forces data for each protein"""
        output_path = os.path.join(self.save_path, pdbid)

        # Remove legacy prior files if they exist
        if os.path.exists(f"{output_path}/raw/{pdbid}_priors.yaml"):
            os.unlink(f"{output_path}/raw/{pdbid}_priors.yaml")
        if os.path.exists(f"{output_path}/raw/{pdbid}_prior_params.json"):
            os.unlink(f"{output_path}/raw/{pdbid}_prior_params.json")

        # Generate delta forces for all atom simulation vs. prior FF
        coords_npz = f'{output_path}/raw/coordinates.npy'
        forces_npz = f'{output_path}/raw/forces.npy'
        delta_forces_npz = f'{output_path}/raw/deltaforces.npy'
        prior_energy_npz = f'{output_path}/raw/prior_energy.npy'
        box_npz = None
        if self.box:
            box_npz = f"{output_path}/raw/box.npy"
        forcefield = os.path.join(self.save_path, "priors.yaml")
        psf_file = f'{output_path}/processed/{pdbid}_processed.psf'
        prior_params = self.prior_builder.prior_params
        
        deltaForcesObj = DeltaForces(DEVICE_STEP_3, psf_file, coords_npz, box_npz)
        if 'external' in self.prior_builder.prior_params.keys():
            # forceterms = ['bonds', 'angles', 'dihedrals']
            deltaForcesObj.addExternalForces(forcefield, self.prior_builder.priors['bonds'], self.prior_builder.priors['angles'], self.prior_builder.priors['dihedrals'], forceterms=prior_params["forceterms_nn"], bar_position=bar_position)

            # forceterms = ['repulsioncg'] # update them properly in preprocess.py in the _flex class
            deltaForcesObj.computePriorForces(forcefield, exclusions=prior_params["exclusions"],
                forceterms=prior_params["forceterms_classical"], bar_position=bar_position)

        else:
            deltaForcesObj.computePriorForces(forcefield, exclusions=prior_params["exclusions"],
                forceterms=prior_params["forceterms"], bar_position=bar_position)

        # load MD forces from forces_npz, compute delta forces, and save them in delta_forces_npz
        deltaForcesObj.makeAndSaveDeltaForces(forces_npz, delta_forces_npz, prior_energy_npz) 

def gen_input_mapping(conf):
    """Find the list of input files for the passed dataset config"""
    pdbid_mapping = dict()
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
                    print(f"  Skipping \"{dir_name}\" (directory contains no output)")
    return pdbid_mapping

prior_types = {
    "CA":Prior_CA,
    "CACB":Prior_CACB,
    "CACB_lj":Prior_CACB_lj,
    "CACB_lj_angle_dihedral":Prior_CACB_lj_angle_dihedral,
    "CA_lj":Prior_CA_lj,
    "CA_lj_angle":Prior_CA_lj_angle,
    "CA_lj_angle_dihedral":Prior_CA_lj_angle_dihedral,
    "CA_lj_angle_dihedralX":Prior_CA_lj_angle_dihedralX,
    "CA_lj_angleXCX_dihedralX":Prior_CA_lj_angleXCX_dihedralX,
    "CA_lj_angleXCX_dihedralX_flex":Prior_CA_lj_angleXCX_dihedralX_flex,
    "CA_lj_angleXCX_dihedralX_V1":Prior_CA_lj_angleXCX_dihedralX_V1,
    "CA_Majewski2022_v0":Prior_CA_Majewski2022_v0,
    "CA_Majewski2022_v1":Prior_CA_Majewski2022_v1,
    "CA_lj_bondNull_angleXCX_dihedralX":Prior_CA_lj_bondNull_angleXCX_dihedralX,
    "CA_lj_bondNull_angleNull_dihedralX":Prior_CA_lj_bondNull_angleNull_dihedralX,
    "CA_lj_bondNull_angleNull_dihedralNull":Prior_CA_lj_bondNull_angleNull_dihedralNull,
    "CA_lj_angleNull_dihedralX":Prior_CA_lj_angleNull_dihedralX,
    "CA_lj_angleNull_dihedralNull":Prior_CA_lj_angleNull_dihedralNull,
    "CA_null":Prior_CA_null,
    "CA_lj_only":Prior_CA_lj_only,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("input", nargs = "+", help="Input directory path")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument("--pdbids", nargs="*", help="List of specific PDB IDs to process")
    parser.add_argument("--num-frames", "--num_frames", type=int, default=None, help="Number of frames to process")
    parser.add_argument("--frame-slice", type=str, default=None, help="Select frames to process using a python slice: start:end:stride")
    parser.add_argument("--temp", type=int, default=300, help="Temperature")
    parser.add_argument("--prior", type=str, default=None, help="Select the prior forcefield to use, must be one of: " + ", ".join(sorted(prior_types.keys())))
    parser.add_argument("--optimize-forces", action="store_true", help="Use statistically optimal force aggregation (Kramer 2023)")
    parser.add_argument("--prior-file", default=None, help="Use PRIOR_FILE instead of fitting a prior")
    parser.add_argument('--no-box', default=False, action='store_true', help="Don't use periodic box information")
    parser.add_argument('--prior-plots', default=True, action='store_true', help="Save plots of the prior fit functions")
    parser.add_argument('--no-prior-plots', dest='prior_plots', action='store_false', help="Don't save plots of the prior fit functions")
    parser.add_argument('--no-fit-constraints', default=False, action='store_true', help="Disable range constraints when fitting prior functions")
    parser.add_argument('--fit-min-cnt', type=int, default=0, help="Only bins with cnt > min_cnt will be considered when fitting the prior (default 0)")
    # parser.add_argument('--tag-beta-turns', default=False, action='store_true', help="Give beta turns a different bond type in the prior")
    parser.add_argument('--resume', default=False, action='store_true', help="Resume processing rather than overwriting, all settings must be identical between calls")
    parser.add_argument('--num-cores', type=int, default=32, help="Number of cores to be used for parallelization of preprocessing")
    parser.add_argument('--jobid', type=int, default=None, help="Integer denoting jobid, if not -1, it will only process a subset of the PDBs")
    parser.add_argument('--totalNrJobs', type=int, default=None, help="Integer denoting how many jobs are in total.")
    

    args = parser.parse_args()
    print(args)

    output_dir = args.output
    pdbids = args.pdbids
    assert not (args.num_frames and args.frame_slice)
    if args.num_frames:
        frame_slice = slice(0, args.num_frames)
    elif args.frame_slice:
        # Convert the arg string into a slice
        frame_slice = slice(*[int(i) if i != '' else None for i in args.frame_slice.split(":") ])
    else:
        frame_slice = slice(None)
    temp = args.temp
    optimize_forces = args.optimize_forces
    box = not args.no_box
    prior_plots = args.prior_plots
    prior_name = args.prior
    prior_file = args.prior_file
    resume_preprocess = args.resume
    num_cores = args.num_cores
    jobid = args.jobid
    totalNrJobs = args.totalNrJobs

    if prior_file:
        assert os.path.exists(prior_file), f"Prior file does not exist: {prior_file}"
        prior_params_path = get_prior_params_path(prior_file)
        with open(prior_params_path, "r", encoding="utf-8") as f:
            prior_params = json.load(f)
            prior_configuration_name = prior_params["prior_configuration_name"]
            if prior_name is None:
                prior_name = prior_configuration_name
            elif prior_name != prior_configuration_name:
                print()
                print(f"WARNING: Prior \"{prior_name}\" differs from the one used to build the prior file \"{prior_configuration_name}\"")
                print()

    assert prior_name, " You must specify the prior to use with either --prior or --prior-file"

    if prior_name not in prior_types:
        raise RuntimeError(f"Unknown prior configuration: {prior_name}")
    print(f"Using prior: {prior_name}")
    prior_builder = prior_types[prior_name]() # <- () to instantiate the class
    prior_builder.enable_fit_constraints(not args.no_fit_constraints)
    # prior_builder.enable_bond_tags(args.tag_beta_turns)
    prior_builder.enable_bond_tags(False)
    prior_builder.set_min_cnt(args.fit_min_cnt)

    if 'external' in prior_builder.prior_params.keys():
        mp.set_start_method('spawn')

    # Set matplotlib to use a thread safe backend (for prior fit plots)
    import matplotlib
    matplotlib.use('Agg')

    dataset_conf = []

    for i in args.input:
        if os.path.isfile(i):
            with open(args.input[0], "r") as f:
                dataset_conf += yaml.safe_load(f)
        else:
            dataset_conf += [{"path": i}]

    input_path_map = gen_input_mapping(dataset_conf)

    if pdbids:
        input_path_map = {i: input_path_map[i] for i in pdbids}

    preprocessor = Preprocessor(dataset_conf, input_path_map, output_dir, prior_builder, prior_file, prior_name, frame_slice, temp, optimize_forces, box, prior_plots, resume_preprocess, num_cores, jobid, totalNrJobs)

    preprocessor.preprocess()

