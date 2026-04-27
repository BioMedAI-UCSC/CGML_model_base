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

from .mapping import CGMappingDef_CA, CGMappingDef_CACB, CGMappingDef_CA_DNA

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


class Prior_CA_DNA(Prior_CA):
    """Protein Cα + DNA 2-bead model; after bonded/angle fit, applies universal DNA stiffness (cgschnet)."""

    def __init__(self) -> None:
        super().__init__()
        self.prior_params.update(
            {
                "prior_configuration_name": "CA_DNA",
                "exclusions": ["1-2", "1-3"],
                "forceterms": ["bonds", "angles"],
            }
        )
        self.terms["bonds"] = prior.ParamBondedCalculator()
        self.terms["angles"] = prior.ParamAngleCalculator(center=False)
        self.cg_mapping_def: CGMappingDef_CA_DNA = CGMappingDef_CA_DNA()

    def init_prior_dict(self) -> None:
        super().init_prior_dict()
        bead_type_to_mass: dict[str, float] = {}
        for resname in self.cg_mapping_def.bead_types:
            bead_types = self.cg_mapping_def.bead_types[resname]
            bead_masses = self.cg_mapping_def.bead_masses.get(resname, [12.01] * len(bead_types))
            for bt, bm in zip(bead_types, bead_masses):
                bead_type_to_mass[bt] = float(bm)
        for atom_type in self.priors["atomtypes"]:
            if atom_type in bead_type_to_mass:
                self.priors["masses"][atom_type] = bead_type_to_mass[atom_type]

    def fit(
        self,
        temperature: float,
        plot_dir: str | None = None,
        use_cached_fits: list | None = None,
    ) -> None:
        if use_cached_fits is None:
            use_cached_fits = []
        print("[DEBUG] Starting fit for Prior_CA_DNA")
        super().fit(temperature, plot_dir, use_cached_fits)
        self._apply_universal_cg_parameters()
        self._ensure_all_dna_params_exist()
        self._validate_physics()
        print("[DEBUG] Applied universal CG parameters for mixed protein–DNA systems")

    def _apply_universal_cg_parameters(self) -> None:
        UNIVERSAL = {
            "DNA_BOND_K": 50.0,
            "DNA_ANGLE_K": 30.0,
            "PROTEIN_BOND_K": 10.0,
            "PROTEIN_ANGLE_K": 2.0,
            "DNA_BACKBONE_LENGTH": 5.5,
            "DNA_BASE_DISTANCE": 5.75,
            "DNA_BACKBONE_ANGLE": 170.0,
        }
        if "bonds" in self.priors:
            for bond_type, params in self.priors["bonds"].items():
                if isinstance(params, dict):
                    current_k = float(params.get("k0", 1.0))
                    current_r0 = float(params.get("req", 3.8))
                elif isinstance(params, (list, tuple)) and len(params) >= 2:
                    current_k = float(params[0])
                    current_r0 = float(params[1])
                else:
                    print(f"[WARNING] Bond {bond_type} has unexpected format: {params}")
                    continue
                dna_token = any(t in bond_type for t in ("DBB", "DBA", "DBT", "DBG", "DBC"))
                if dna_token:
                    new_k = UNIVERSAL["DNA_BOND_K"]
                    if "DBB-DBB" in bond_type:
                        new_r0 = UNIVERSAL["DNA_BACKBONE_LENGTH"]
                    elif "DBB" in bond_type:
                        new_r0 = UNIVERSAL["DNA_BASE_DISTANCE"]
                    else:
                        new_r0 = current_r0
                    self.priors["bonds"][bond_type] = {"k0": float(new_k), "req": float(new_r0)}
                elif "CA" in bond_type and current_k < 5.0:
                    if isinstance(self.priors["bonds"][bond_type], dict):
                        self.priors["bonds"][bond_type]["k0"] = UNIVERSAL["PROTEIN_BOND_K"]
                    else:
                        self.priors["bonds"][bond_type][0] = UNIVERSAL["PROTEIN_BOND_K"]

        if "angles" in self.priors:
            for angle_type, value in self.priors["angles"].items():
                if not isinstance(value, dict):
                    continue
                try:
                    current_k0 = float(value.get("k0", 1.0))
                except (ValueError, TypeError):
                    continue
                dna_kw = ("DBA", "DBB", "DBT", "DBG", "DBC")
                if any(kw in angle_type for kw in dna_kw):
                    if "DBB-DBB-DBB" in angle_type:
                        value["theta0"] = UNIVERSAL["DNA_BACKBONE_ANGLE"]
                        value["k0"] = UNIVERSAL["DNA_ANGLE_K"]
                    else:
                        value["k0"] = UNIVERSAL["DNA_ANGLE_K"]
                elif "CA" in angle_type and current_k0 < 0.5:
                    value["k0"] = UNIVERSAL["PROTEIN_ANGLE_K"]

    def _has_dna_beads(self) -> bool:
        """True if any loaded molecule has DNA CG atom types (DB*), not just protein Cα."""
        for at in self.atom_types:
            s = str(at)
            if s.startswith("DB") or s.startswith("db"):
                return True
        return False

    def _ensure_all_dna_params_exist(self) -> None:
        if not self._has_dna_beads():
            return
        essential_dna_bonds = {
            "DBB-DBB": [50.0, 5.5],
            "DBB-DBA": [50.0, 5.75],
            "DBB-DBT": [50.0, 5.75],
            "DBB-DBG": [50.0, 5.75],
            "DBB-DBC": [50.0, 5.75],
        }
        essential_dna_angles = {
            "DBB-DBB-DBB": {"k0": 30.0, "theta0": 170.0},
            "DBA-DBB-DBB": {"k0": 25.0, "theta0": 50.0},
            "DBB-DBB-DBC": {"k0": 25.0, "theta0": 83.0},
            "DBB-DBB-DBG": {"k0": 25.0, "theta0": 78.0},
            "DBB-DBB-DBT": {"k0": 25.0, "theta0": 55.0},
        }
        for bond_type, params in essential_dna_bonds.items():
            if bond_type not in self.priors.get("bonds", {}):
                self.priors.setdefault("bonds", {})[bond_type] = params
        for angle_type, params in essential_dna_angles.items():
            if angle_type not in self.priors.get("angles", {}):
                self.priors.setdefault("angles", {})[angle_type] = params

    def _validate_physics(self) -> None:
        print("\n[PHYSICS VALIDATION]")
        print("=" * 50)
        dna_stiffness_ok = True
        if not self._has_dna_beads():
            print(" (no DNA CG beads in this batch — skip DNA stiffness template checks)\n" + "=" * 50)
            return
        for bond_type, params in self.priors.get("bonds", {}).items():
            if not any(t in bond_type for t in ("DBB", "DBA", "DBT", "DBG", "DBC")):
                continue
            try:
                if isinstance(params, dict):
                    k = float(params.get("k0", 0))
                    r0 = float(params.get("req", 0))
                elif isinstance(params, (list, tuple)) and len(params) >= 2:
                    k, r0 = float(params[0]), float(params[1])
                else:
                    print(f" DNA bond {bond_type}: Unrecognized format {params}")
                    continue
                if k < 20.0:
                    print(f"  DNA bond {bond_type}: k={k:.1f} (might be too soft)")
                    dna_stiffness_ok = False
                if r0 < 4.0 or r0 > 7.0:
                    print(f"  DNA bond {bond_type}: r0={r0:.1f}Å (unusual)")
            except (ValueError, TypeError, IndexError):
                print(f"Could not parse DNA bond {bond_type}: {params}")
        avg_dna_k = 0.0
        avg_p_k = 0.0
        dna_c = 0
        p_c = 0
        for bond_type, params in self.priors.get("bonds", {}).items():
            try:
                if isinstance(params, dict):
                    k = float(params.get("k0", 0))
                elif isinstance(params, (list, tuple)) and len(params) >= 2:
                    k = float(params[0])
                else:
                    continue
            except (ValueError, TypeError, IndexError):
                continue
            if any(t in bond_type for t in ("DBB", "DBA", "DBT", "DBG", "DBC")):
                avg_dna_k += k
                dna_c += 1
            elif "CA" in bond_type:
                avg_p_k += k
                p_c += 1
        if dna_c > 0 and p_c > 0:
            avg_dna_k /= dna_c
            avg_p_k /= p_c
            ratio = avg_dna_k / avg_p_k if avg_p_k > 0 else 999.0
            print(f" Stiffness ratio (DNA:Protein) = {ratio:.1f}:1")
            print(f" Average DNA k = {avg_dna_k:.1f}, Protein k = {avg_p_k:.1f}")
            if ratio < 2.0:
                print("DNA not stiff enough relative to protein!")
        print("\n[ANGLES VALIDATION]")
        dna_ang = 0
        tot = 0.0
        for angle_type, value in self.priors.get("angles", {}).items():
            if not isinstance(value, dict):
                continue
            if not any(t in angle_type for t in ("DBA", "DBB", "DBT", "DBG", "DBC")):
                continue
            try:
                k0 = float(value.get("k0", 0))
                tot += k0
                dna_ang += 1
                if "DBB-DBB-DBB" in angle_type:
                    t0 = float(value.get("theta0", 0))
                    if t0 < 150 or t0 > 190:
                        print(
                            f"DNA backbone angle {angle_type}: theta0={t0:.1f}° (should be ~170°)"
                        )
            except (ValueError, TypeError):
                pass
        if dna_ang > 0:
            print(f"  Average DNA angle k0 = {tot / dna_ang:.1f}")
            if tot / dna_ang < 10.0:
                print("DNA angles might be too soft!")
        if dna_stiffness_ok:
            print("\n  DNA bond parameters look reasonable")
        else:
            print("\n  DNA bond parameters need attention")
        print("=" * 50)

    def build_mapping(self, topology):
        return CGMapping(topology, self.cg_mapping_def)

    def map_embeddings(self, selected_atoms, topology):  # pyright: ignore[reportIncompatibleMethodOverride]
        standard = {
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLU",
            "GLN",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        }
        dna = {"DA", "DT", "DG", "DC"}
        all_r = sorted(standard | dna)
        residue_map = {res: i + 1 for i, res in enumerate(all_r)}
        result: list[int] = []
        for a_idx in selected_atoms:
            r_name = "".join(filter(str.isalpha, topology.atom(a_idx).residue.name))
            if r_name not in residue_map:
                print(f"[WARNING] Residue {r_name!r} not in mapping!")
            result.append(residue_map.get(r_name, 0))
        return np.array(result, dtype=int)

    def write_psf(self, pdb_file, psf_file):
        bonds = "bonds" in self.terms
        angles = "angles" in self.terms
        dihedrals = "dihedrals" in self.terms
        return psfwriter.pdb2psf_CA(
            pdb_file,
            psf_file,
            bonds=bonds,
            angles=angles,
            dihedrals=dihedrals,
            tag_beta_turns=self.tag_beta_turns,
        )


PRIOR_TYPES = {
    "CA": Prior_CA,
    "CA_DNA": Prior_CA_DNA,
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
