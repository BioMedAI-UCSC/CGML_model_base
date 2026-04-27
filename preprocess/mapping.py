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


class CGMappingDef_CA_DNA(CGMappingDef_CA):
    """Protein Cα plus coarse DNA (2 beads/residue: backbone + base) — matches cgschnet `Prior_CA_DNA` mapping."""

    def __init__(self) -> None:
        super().__init__()
        dna_residues = ["DA", "DT", "DG", "DC"]
        backbone_atoms = [
            "P",
            "OP1",
            "OP2",
            "O5'",
            "C5'",
            "C4'",
            "C3'",
            "O3'",
            "C1'",
            "C2'",
            "O4'",
        ]
        # 5' or fragment models without phosphate: COM of sugar + backbone linkage only (same order as cgschnet, minus P/OP*)
        backbone_atoms_no_phosphate = [
            "O5'",
            "C5'",
            "C4'",
            "C3'",
            "O3'",
            "C1'",
            "C2'",
            "O4'",
        ]
        base_atoms = {
            "DA": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
            "DT": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "C7"],
            "DG": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
            "DC": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
        }
        for resname in dna_residues:
            self.bead_atom_selection[resname] = [backbone_atoms, base_atoms[resname]]
        # Tried in order for the first (DBB) bead; see `module.cg_mapping.CGMapping`
        self.dna_backbone_atom_candidates = (backbone_atoms, backbone_atoms_no_phosphate)
        for resname in dna_residues:
            base_code = resname[1]
            self.bead_atom_names[resname] = ["DBB", f"DB{base_code}"]
        for resname in dna_residues:
            base_code = resname[1]
            self.bead_types[resname] = ["DBB", f"DB{base_code}"]
        base_masses = {"A": 134.1, "T": 125.1, "G": 150.1, "C": 110.1}
        for resname in dna_residues:
            base_code = resname[1]
            self.bead_masses[resname] = [178.08, base_masses[base_code]]
        current_max_id = max(self.bead_embeddings.values())[0]
        dbb_id = current_max_id + 1
        base_ids = {"A": dbb_id + 1, "T": dbb_id + 2, "G": dbb_id + 3, "C": dbb_id + 4}
        for resname in dna_residues:
            base_code = resname[1]
            self.bead_embeddings[resname] = [dbb_id, base_ids[base_code]]
        for resname in dna_residues:
            self.bead_backbone_idx[resname] = 0
