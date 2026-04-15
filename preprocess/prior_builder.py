"""Prior builders and registry for CG preprocessing."""

import json
import os
import pickle

import numpy as np
import yaml

from module import prior
from module import prior_flex
from module import psfwriter
from module.cg_mapping import CGMapping
from module.torchmd_cg_mappings import CACB_MAP

from .mapping import CGMappingDef_CA, CGMappingDef_CACB

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

    def fit(self, temperature, plot_dir=None, use_cached_fits=None):
        if use_cached_fits is None:
            use_cached_fits = []
        self.init_prior_dict()
        assert self.priors is not None
        for key, term in self.terms.items():
            cache_pkl = os.path.join(plot_dir, f"prior_{key}.pkl") if plot_dir else None
            if cache_pkl and os.path.exists(cache_pkl) and (key in use_cached_fits):
                print(f"Used cached fit for {key}...")
                with open(cache_pkl, "rb") as f:
                    self.priors[key] = pickle.load(f)
            else:
                print(f"Fitting {key}...")
                self.priors[key] = term.get_param(temperature, plot_dir, self.fit_constraints, self.min_cnt)
                if plot_dir:
                    with open(os.path.join(plot_dir, f"prior_{key}.pkl"), "wb") as f:
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

PRIOR_TYPES = {
    "CA": Prior_CA,
    "CACB": Prior_CACB,
    "CACB_lj": Prior_CACB_lj,
    "CACB_lj_angle_dihedral": Prior_CACB_lj_angle_dihedral,
    "CA_lj": Prior_CA_lj,
    "CA_lj_angle": Prior_CA_lj_angle,
    "CA_lj_angle_dihedral": Prior_CA_lj_angle_dihedral,
    "CA_lj_angle_dihedralX": Prior_CA_lj_angle_dihedralX,
    "CA_lj_angleXCX_dihedralX": Prior_CA_lj_angleXCX_dihedralX,
    "CA_lj_angleXCX_dihedralX_flex": Prior_CA_lj_angleXCX_dihedralX_flex,
    "CA_lj_angleXCX_dihedralX_V1": Prior_CA_lj_angleXCX_dihedralX_V1,
    "CA_Majewski2022_v0": Prior_CA_Majewski2022_v0,
    "CA_Majewski2022_v1": Prior_CA_Majewski2022_v1,
    "CA_lj_bondNull_angleXCX_dihedralX": Prior_CA_lj_bondNull_angleXCX_dihedralX,
    "CA_lj_bondNull_angleNull_dihedralX": Prior_CA_lj_bondNull_angleNull_dihedralX,
    "CA_lj_bondNull_angleNull_dihedralNull": Prior_CA_lj_bondNull_angleNull_dihedralNull,
    "CA_lj_angleNull_dihedralX": Prior_CA_lj_angleNull_dihedralX,
    "CA_lj_angleNull_dihedralNull": Prior_CA_lj_angleNull_dihedralNull,
    "CA_null": Prior_CA_null,
    "CA_lj_only": Prior_CA_lj_only,
}
