"""CG bead mapping definitions."""

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
