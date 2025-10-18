#!/usr/bin/env python3

import os
import glob
import numpy as np
import traceback
import yaml
import torch
import json
import time
import itertools
# from torchmdnet.models.model import create_model
from module.torchmdnet.model import create_model
from moleculekit.molecule import Molecule
# from torchmd.forcefields.forcefield import ForceField
from module.torchmd import tagged_forcefield
from torchmd.forces import Forces
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmd.integrator import maxwell_boltzmann
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper
from torchmd.minimizers import minimize_bfgs
from tqdm import tqdm
import mdtraj
from mdtraj.formats import PDBTrajectoryFile, HDF5TrajectoryFile
import tempfile
import textwrap
import pickle
from module import dataset
from module import model_util
from module import torchforcefield

from module.make_deltaforces import ExternalNN, ParametersNN
from preprocess import Prior_CA_lj_angleXCX_dihedralX_flex

class CalcWrapper():
    def __init__(self, ff, models):
        self.ff = ff
        self.models = models
        self.par = ff.par

    def compute(self, coords, box, forces_out):
        prior_energy = self.ff.compute(coords, box, forces_out)
        total_energy = prior_energy.detach().cpu().flatten()
        for model in self.models:
            model_energy, model_forces, _ = model.calculate(coords, box)
            forces_out.add_(model_forces)
            total_energy += model_energy.detach().cpu().flatten()
        
        assert len(total_energy) == len(coords)
        return total_energy

def can_compile():
    version_major, version_minor = torch.__version__.split(".")[:2]
    if not (int(version_major) >= 2 and int(version_minor) >= 4):
        return False
    try:
        import triton # type: ignore
    except Exception as _:
        return False
    return True

""" 
buffer - tensor allocated on gpu. 
input and output buffers have to be pre-allocated and become part of the graph 

CUDAStream - series of CUDA kernels that are being executed

this is running only the prior, not the main net yet!
"""
def graph_forward(module, box, data, repeats=20):
    # detach().clone() is nedded here or the model results (not the prior) get corrupted somehow - Daniel 2025.04.26
    static_in = torch.as_tensor(data.detach().clone(), device=module.device)
    if box is not None and not torch.all(box == 0):
        static_box = torch.ones((3,), device=module.device)
    else:
        static_box = None
    static_out = torch.zeros_like(data, device=module.device)
    static_pots_out = torch.zeros(1, device=module.device)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream()) #pyright: ignore[reportAttributeAccessIssue]
    with torch.cuda.stream(s): # the commands within the "with" block happen in that stream # pyright: ignore[reportArgumentType]
        for _ in range(repeats): # have to run it several times until CUDA stops changing the buffers
             # module is the entire network, so the call below runs the forward pass through the entire network
             # important: the module.forward() MUST!! be deterministic.
            static_pots_out[:] = module.forward(static_in, static_box, static_out)
    torch.cuda.current_stream().wait_stream(s)

    # now we'll create a graph we wanna record
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): # everything inside here gets recorded in the graph
        # important: the module.forward() MUST!! be deterministic.
        static_pots_out[:] = module.forward(static_in, static_box, static_out)

    def eval_graph(co, box, fo_out):
        # co.requires_grad = False
        static_in.copy_(co)
        if static_box is not None:
            static_box.copy_(box)
        g.replay()
        fo_out.copy_(static_out)
        return static_pots_out

    module.forward = eval_graph


class External():
    def __init__(self, model, embeddings, device, num_replicates, sequence=None, bonds=None, angles=None, dihedrals=None):
        self.model = model
        self.device = device
        self.n_atoms = len(embeddings)
        self.use_box = True

        e = np.array(embeddings)
        b = np.array([i for i in range(0,num_replicates)])
        batch_nums = np.repeat(b, len(embeddings), axis=0).flatten()
        embeddings = np.concatenate(np.repeat(e[np.newaxis,:], num_replicates, axis=0))
        self.batch_nums = torch.tensor(batch_nums, dtype=torch.long)
        self.embeddings = torch.tensor(embeddings)

        self.batch_nums = self.batch_nums.to(device)
        self.embeddings = self.embeddings.to(device)

        if sequence is not None:
            sequence = np.repeat(sequence[np.newaxis,:], num_replicates, axis=0).flatten()
            self.sequence = torch.tensor(sequence, dtype=torch.long, device=device)
        else:
            self.sequence = None

        self.bonds = None
        if bonds is not None:
            bonds = np.concatenate([bonds + i*self.n_atoms for i in range(0,num_replicates)])
            self.bonds = torch.as_tensor(bonds, dtype=torch.long, device=device)

        self.angles = None
        if angles is not None:
            angles = np.concatenate([angles + i*self.n_atoms for i in range(0,num_replicates)])
            self.angles = torch.as_tensor(angles, dtype=torch.long, device=device)

        self.dihedrals = None
        if dihedrals is not None:
            dihedrals = np.concatenate([dihedrals + i*self.n_atoms for i in range(0,num_replicates)])
            self.dihedrals = torch.as_tensor(dihedrals, dtype=torch.long, device=device)

        # Energy is in (kcal/mol), forces in (kcal/Ang/mol)
        self.output_transformer = lambda energy, forces: (
                energy,
                forces,
                None
            )

    # From torchmd.calculators
    def calculate(self, pos, box):
        #TODO: Bring back the cuda graph stuff from torchmd.calculators?

        # The model expects all the replicate positions to be flattened into one array
        # Each replicate is distinguished by the self.batch_nums calculated above
        pos = pos.to(self.device).type(torch.float32).reshape(-1, 3)

        # If there's no box we get passed an array of zeros, but the model expects to see None in this case
        if not self.use_box:
            box = None

        kwargs = {}
        if self.bonds is not None:
            kwargs["bonds"] = self.bonds
        if self.angles is not None:
            kwargs["angles"] = self.angles
        if self.dihedrals is not None:
            kwargs["dihedrals"] = self.dihedrals

        self.energy, self.forces = self.model(self.embeddings, pos, self.batch_nums, box=box, s=self.sequence, **kwargs)[:2]

        assert self.forces is not None, "The model is not returning forces"
        assert self.energy is not None, "The model is not returning energy"

        return self.output_transformer(
            self.energy.clone().detach(),
            self.forces.clone().reshape(-1, self.n_atoms, 3).detach(),
        )

def glob_one(glob_str):
    """Perform a glob match and assert that it only matched a single file"""
    path = glob.glob(glob_str)
    assert len(path) == 1, f"Ambiguous file selection: {glob_str}"
    return path[0]

def write_pdb_title(pdb_writer, strings):
    # https://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#TITLE
    # The title can be multiple lines, if a line is too long it can be wrapped with a continuation
    for line in strings.split("\n"):
        # String field width is 70 (columns 11 to 80)
        wrapped = textwrap.wrap(line, 70)
        for i, string in enumerate(wrapped):
            if i == 0:
                cont = ""
            else:
                cont = str(i+1)
            pdb_writer._file.write(f"TITLE   {cont:2}{string}\n")

def load_model(checkpoint_path, device, hyper_params=None, max_num_neighbors=None, extra_model_config={}, verbose=True):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not hyper_params:
        hyper_params = checkpoint["hyper_parameters"]
    if max_num_neighbors:
        hyper_params["max_num_neighbors"] = max_num_neighbors
    # Patch the config with the values in extra_model_config
    # TODO: Handle max_num_neighbors here too?
    for k, v in extra_model_config.items():
        hyper_params[k] = v

    model = create_model(args=hyper_params)
    model_util.load_state_dict_with_rename(model, checkpoint["state_dict"])
    model.to(device)

    if verbose:
        print("--- Model ---\n",model,"-------------\n")

    return model

def load_molecule(prior_path, prior_params, processed_path, use_box, verbose=True):
    if os.path.isdir(processed_path):
        protein_path = os.path.join(processed_path, "processed/topology.psf")
        if not os.path.exists(protein_path):
            protein_path = glob_one(os.path.join(processed_path, "processed/*_processed.psf"))
        coords_path = os.path.join(processed_path, "raw/coordinates.npy")
        embeddings_path = os.path.join(processed_path, "raw/embeddings.npy")
        box_path = os.path.join(processed_path, "raw/box.npy")

        print("Topology:   ",protein_path)
        print("Prior:      ",prior_path)
        print("Coordinates:",coords_path)
        print("Embeddings: ",embeddings_path)
        if use_box and os.path.exists(box_path):
            print("Box:        ", box_path)
        elif not use_box:
            print("Box:        ", "<disabled>")
        else:
            use_box = False
            print("Box:        ", "<none>")

        mol = Molecule(protein_path)
        coords = np.load(coords_path)
        print("Shape", coords.shape)
        embeddings = np.load(embeddings_path)
        if use_box:
            box = np.load(box_path)
            mol.box = np.diag(box[0]).reshape(3,1)
    else:
        import preprocess
        prior_name = prior_params["prior_configuration_name"]
        prior_builder = preprocess.prior_types[prior_name]()
        if verbose:
            print("Prior Config:", prior_name)
            print("Structure:   ", processed_path)

        traj = mdtraj.load_frame(processed_path, 0)
        traj.center_coordinates()

        cg_map = prior_builder.build_mapping(traj.topology)
        coords = cg_map.cg_positions(traj.xyz * 10)
        embeddings = np.array(cg_map.embeddings)
        mol = cg_map.to_mol(bonds=True, angles=True, dihedrals=True)

        mol.box = np.zeros((3, 0), dtype=np.float32)
        if use_box and (traj.unitcell_lengths is not None):
            mol.box = np.diag(traj.unitcell_vectors[0] * 10).reshape(3,1)
            if verbose:
                print("Box:         ", "<from structure>")
        elif not use_box:
            if verbose:
                print("Box:         ", "<disabled>")
        else:
            if verbose:
                print("Box:         ", "<none>")

    # MoleculeKit puts frames in the last dimension
    mol.coords = coords[0].reshape(-1, 3, 1)
    if verbose:
        print("Mol:", mol.coords.shape, mol.frame, mol.box.flatten() if mol.box is not None else None)
    return mol, embeddings

def load_molecules(prior_path, prior_params, starting_pos_paths: list[str], use_box, verbose=True):
    loaded_molecules = [load_molecule(prior_path, prior_params, path, use_box, verbose) for path in starting_pos_paths]
    mols = [x[0] for x in loaded_molecules]

    # Assert that all loaded molecules have the same topology
    first_embedding = loaded_molecules[0][1]
    for molecule in loaded_molecules:
        assert (loaded_molecules[0][1] == molecule[1]).all()
        assert (loaded_molecules[0][0].angles == molecule[0].angles).all()
        assert (loaded_molecules[0][0].bonds == molecule[0].bonds).all() #pyright: ignore[reportAttributeAccessIssue]
        assert (loaded_molecules[0][0].dihedrals == molecule[0].dihedrals).all()

    return mols, first_embedding

def make_system(mols, prior_path, calcs, device, forceterms, exclusions, replicas, temperature=300, new_ff=False):
    precision = torch.float
    use_box = False

    system = System(mols[0].numAtoms, replicas, precision, device)
    system.set_positions(np.concatenate([x.coords for x in mols], axis=2))
    if mols[0].box.size > 0:
        system.set_box(np.concatenate([x.box for x in mols], axis=1))
        use_box = True

    for calc in calcs:
        calc.use_box = use_box

    if new_ff:
        forces = torchforcefield.TorchForceField(prior_path, mols[0], device=device, cutoff=None,
                                             terms=forceterms,
                                             exclusions=exclusions,
                                             use_box=use_box)

        forces = CalcWrapper(forces, calcs)

        if can_compile():
            t0 = time.time()
            print("Compiling prior...", end="", flush=True)
            forces.ff.forward = torch.compile(forces.ff.forward)
            print(f" Done ({time.time() - t0:.2f}s)")
        else:
            print("Skipping prior compile, requirements not met.")
        t0 = time.time()
        print("Building CUDA graph...", end="", flush=True)
        graph_forward(forces.ff, system.box[0], system.pos[0])
        print(f" Done ({time.time() - t0:.2f}s)")

    else:
        # Use the first toplology to initialize the forcefield
        ff = tagged_forcefield.create(mols[0], prior_path)

        parameters = Parameters(ff, mols[0], forceterms, precision=precision, device=device)

        
        forces = Forces(
            parameters,
            terms=forceterms,
            external=calc, #pyright: ignore[reportPossiblyUnboundVariable]
            cutoff=None, # Default none
            rfa=False, # Taken from Chingolin example yaml
            switch_dist=None, # Default none
            exclusions=exclusions,
        )

    system.set_velocities(
        maxwell_boltzmann(forces.par.masses, temperature, replicas)
    )

    return system, forces

# From torchmd/run.py
def dynamics(args, mol, system, forces, device, title=""):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # device = torch.device(args.device)
    integrator = Integrator(
        system,
        forces,
        args.timestep,
        device,
        gamma=args.langevin_gamma,
        T=args.langevin_temperature,
    )
    wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None, device)

    if args.minimize is not None:
        minimize_bfgs(system, forces, steps=args.minimize)

    iterator = tqdm(range(1, int(args.steps / args.output_period) + 1), dynamic_ncols=True)

    system.pos.requires_grad = False
    # The integrator expects the forces to be valid before the first step() is called
    Epot = forces.compute(system.pos, system.box, system.forces)

    trajEpot = []
    trajEkin = []
    trajTemp = []
    trajTime = []
    trajPos  = []
    i = 0
    try:
        for i in iterator:
            Ekin, Epot, T = integrator.step(niter=args.output_period)
            wrapper.wrap(system.pos, system.box)
            currpos = system.pos.detach().cpu().numpy().copy()

            trajEkin.append(Ekin)
            trajEpot.append(Epot)
            trajTemp.append(T)
            trajTime.append(np.repeat(i*args.timestep, args.replicas))
            trajPos.append(currpos)

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        print(f"Exception occurred on on iteration {i}, frame >= {i*args.output_period}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        # We need a topology, but mdtraj says the psf file is invalid,
        # moleculekit will open the psf but can't save multi-frame pdbs
        # NOTE: The psf files created by more recent versions of preprocess.py should be valid, but
        #       we can keep this code for now to run the output of older models - Daniel 2024.06.23
        topology_path = os.path.join(tmpdirname, "topology.pdb")
        mol.write(topology_path)
        topology = mdtraj.load(topology_path).top

    file_name, file_ext = os.path.splitext(args.o)
    if not file_ext:
        file_ext = ".pdb"

    for i in range(0, args.replicas):
        try:
            if file_ext == ".pdb":
                writeTraj = PDBTrajectoryFile(f"{file_name}_{i}.pdb", mode="w")

                if title:
                    write_pdb_title(writeTraj, title)

                for j, f in enumerate(trajPos):
                    # Most mdtraj functions assume nanometers, however the PDBTrajectoryFile.write() function
                    # does no unit conversions and expects angstroms?
                    writeTraj.write(f[i], topology, modelIndex=j)
            elif file_ext == ".h5":
                writeTraj = HDF5TrajectoryFile(f"{file_name}_{i}.h5", mode="w", compression=None) #pyright: ignore[reportArgumentType]
                writeTraj.topology = topology
                # TODO: Add more metadata?
                writeTraj.title = title
                # TODO: Verify torchmd energy units are kcal/mol
                writeTraj.write(
                    coordinates=np.array([f[i] for f in trajPos])/10,         # A -> nm
                    time=np.array([f[i] for f in trajTime])*1e-3,             # femtoseconds -> picoseconds
                    kineticEnergy=np.array([f[i] for f in trajEkin])*4.184,   # kcal/mol -> kJ/mol
                    potentialEnergy=np.array([f[i] for f in trajEpot])*4.184, # kcal/mol -> kJ/mol
                    temperature=np.array([f[i] for f in trajTemp]),           # Kelvin
                )
            

        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(e)
            print(f"Exception occurred while saving replica {i}")

    # this used to be ".npy" extension when it was actually a pickle file??? -- andy
    # saved one file for all replicas using pickle - this is used by gen_benchmark.py
    if file_ext == ".pkl":
                # divide xyz by 10 to convert A -> nm, and multiply time by 1e-3 to convert fs -> ps
                mdtraj_list = []
                for i in range(args.replicas):
                    coordinates = np.array([f[i] for f in trajPos])/10         # A -> nm
                    time = np.array([f[i] for f in trajTime])*1e-3        # femtoseconds -> picoseconds
                    # unitcells = torch.diagonal(system.box .repeat(coordinates.shape[0], 1)
                    unitcells = system.box[i,:,:].repeat(coordinates.shape[0], 1, 1)
                    
                    t = mdtraj.Trajectory(coordinates, topology, time)
                    t.unitcell_vectors = unitcells.cpu().numpy()
                    mdtraj_list += [t]
                    
                assert len(mdtraj_list) == args.replicas
                with open(f"{file_name}.pkl", mode="wb") as f:
                    print(f'Saving replicas to {file_name}.pkl')
                    pickle.dump(dict(mdtraj_list=mdtraj_list, topology=topology, title=title), f)

def gen_T5_embeddings(molecule, device):
    from transformers import T5EncoderModel, T5Tokenizer #pyright: ignore[reportPrivateImportUsage]

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    # Generate the FASTA sequence for each chain (assumes 1 bead per residue)
    fasta = "".join([i[-1] for i in molecule.atomtype])
    fasta_list = []
    for i in [len(list(i[1])) for i in itertools.groupby(molecule.segid)]:
        fasta_list.append(fasta[:i])
        fasta = fasta[i:]

    print("Generating ProtT5 embeddings for the sequences:", " ".join(fasta_list))

    embedding_list = []
    for fasta in fasta_list:
        fasta = " ".join(fasta)

        token_encoding = tokenizer(fasta, add_special_tokens=True)
        input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

        with torch.no_grad():
            # The model expects a input of shape [batch, max_len]
            embedding_repr = model(input_ids[None,:], attention_mask=attention_mask[None,:])
            # Output has shape [batch, max_len, embedding_len]
            # We also need to trim off the termination token, in the original script this was done with [:s_len] to
            # also trim off the batch padding
            embedding_list.append(embedding_repr.last_hidden_state[0][:-1])

    embedding = torch.cat(embedding_list).cpu().numpy()
    return embedding

def build_calc(model, mol, embeddings, use_box=False, replicas=1, temperature=300, device="cpu"):
    """Build an External wrapper for model and generate any required input tensors"""

    kwargs = {}

    if hasattr(model.representation_model, "sequence_basis_radius") and \
        model.representation_model.sequence_basis_radius != 0:
        print("Generating sequence info...")
        kwargs["sequence"] = dataset.build_sequence_for_mol(mol)

    if hasattr(model.representation_model, "external_embedding_channels") and \
        model.representation_model.external_embedding_channels > 0:
        print("Generating ProtT5 embeddings...")
        embeddings = gen_T5_embeddings(mol, device)

    if hasattr(model, "harmonic_model"):
        print("Generating classical terms...")
        bonds, angles, dihedrals = dataset.build_classical_terms_for_mol(mol, mol)
        kwargs["bonds"] = bonds
        kwargs["angles"] = angles
        kwargs["dihedrals"] = dihedrals

    calc = External(model, embeddings, device, replicas, **kwargs)
    calc.use_box = use_box

    return calc

# I hate to do this, but it's quick
class ArgsMock():
    pass

def main():
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("checkpoint_path", help="The model checkpoint to use")
    arg_parser.add_argument("processed_path", help="One or more input files from which the model simulations will start. Each different input file will be processed as a different simulation replica.", nargs="+")
    arg_parser.add_argument("--conf", default=None, help="The hyperparameters to use instead of those contained in the checkpoint")
    arg_parser.add_argument("--max-num-neighbors", default=None, type=int, help="Override the 'max_num_neighbors' parameter of the model")
    arg_parser.add_argument("--temperature", default=300, type=int, help="Simulation temperature (Kelvin)")
    arg_parser.add_argument("--timestep", default=1, type=int, help="Simulation timestep (femtoseconds)")
    arg_parser.add_argument("--steps", default=10000, type=int, help="The number of frames to simulate")
    arg_parser.add_argument("-o", "--output", default="sim.pdb", help="The output file name (may be a .pdb or .h5 file)")
    arg_parser.add_argument("--save-steps", default=100, type=int, help="Save frames every n steps")
    arg_parser.add_argument("--prior-only", default=False, action='store_true', help="Disable the model and use only the prior forcefield")
    arg_parser.add_argument('--no-box', action='store_true', help='Do not use box information')
    arg_parser.add_argument("--replicas", default=1, type=int, help="The number of simulations running in parallel")
    arg_parser.add_argument("--torchmd", default=False, action='store_true', help="Use TorchMD for the prior instead of TorchForceField")
    arg_parser.add_argument("--verbose", default=True, action='store_true', help="Prints detailed logs")
    arg_parser.add_argument("--prior-nn", default=None, type=str, help="Path to the folder of a neural network prior.")
    

    args = arg_parser.parse_args()
    print(args)

    run_simulation(args.checkpoint_path, args.processed_path, conf=args.conf, max_num_neighbors=args.max_num_neighbors,
                   temperature=args.temperature, timestep=args.timestep, steps=args.steps, output=args.output, save_steps=args.save_steps,
                   prior_only=args.prior_only, no_box=args.no_box, replicas=args.replicas, torchmd=args.torchmd, verbose=args.verbose,
                   prior_nn=args.prior_nn, gpu=0)

def run_simulation(checkpoint_path, processed_path, conf=None, max_num_neighbors=None,
                   temperature=300, timestep=1, steps=10000, output='sim.pdb', save_steps=100,
                   prior_only=False, no_box=False, replicas=1, torchmd=False, verbose=True,
                   prior_nn=None, gpu=0):

    num_input_paths = len(processed_path)
    
    if num_input_paths != 1:
        assert replicas == 1, "TODO multiple replicas if multiple starting pos"

    replicas = replicas if num_input_paths == 1 else num_input_paths
        
    if conf:
        with open(conf, 'r') as file:
            hyper_params = yaml.safe_load(file)
    else:
        hyper_params = None

    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pth")
    else:
        checkpoint_path = checkpoint_path
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Look for a prior included with the model
    prior_path = os.path.join(checkpoint_dir, "priors.yaml")
    if os.path.exists(prior_path):
        prior_params_path = os.path.join(checkpoint_dir, "prior_params.json")
    else:
        # Otherwise look for one from processed_path
        assert os.path.isdir(processed_path), "Preprocessed input is required when the model doesn't include a prior"
        prior_path = glob_one(os.path.join(processed_path, "raw/*_priors.yaml"))
        prior_params_path = glob_one(os.path.join(processed_path, "raw/*_prior_params.json"))

    # Load forcefield terms
    with open(f"{prior_params_path}", 'r') as file:
       prior_params = json.load(file)

    # Ensure output directory exists
    output_dir = os.path.dirname(output)
    if output_dir:
        assert os.path.exists(output_dir), f"Output directory does not exist: {output_dir}"

    use_box = not no_box
    if prior_only:
        title = ["Prior Only " + os.path.normpath(checkpoint_path).split(os.path.sep)[-2]]
    else:
        title = ["CGSim " + os.path.normpath(checkpoint_path).split(os.path.sep)[-2]]
    title.append(f"(steps={steps}, save={save_steps}, temperature={temperature})")
    title = "\n".join(title)

    device = torch.device("cuda:%d" % gpu if torch.cuda.is_available() else "cpu")
    print('Running on device:', device)

    model: None | torch.nn.Module = None
    if prior_only:
        print("Running in prior only mode...")
    else:
        # print('device & count', device, torch.cuda.device_count(), torch.cuda.get_device_name(gpu))
        model = load_model(checkpoint_path, device, hyper_params=hyper_params, max_num_neighbors=max_num_neighbors, verbose=verbose)

    mols, embeddings = load_molecules(prior_path, prior_params, processed_path, use_box, verbose)
    forceterms = prior_params["forceterms"]
    exclusions = prior_params["exclusions"]

    calcs = []
    if not prior_only:
        assert model is not None
        calcs.append(build_calc(model, mols[0], embeddings, use_box=use_box, replicas=replicas,
                                temperature=temperature, device=device))# pyright: ignore[reportArgumentType]

    if prior_nn:
        print("Loading prior neural network...")
        # prior_nn = ExternalNN(prior_nn, device)

        flexprior = Prior_CA_lj_angleXCX_dihedralX_flex()
        flexprior.load_prior_nnets(prior_nn)
        forceterms = flexprior.priors['prior_params']['forceterms']
        forceterms_nn = flexprior.priors['prior_params']['forceterms_nn']
        forceterms_classical = flexprior.priors['prior_params']['forceterms_classical']
        
        print('loading forceterms', forceterms)
        print('forceterms_nn', forceterms_nn)
        print('forceterms_classical', forceterms_classical)
        parameters = ParametersNN(mols[0], forceterms, precision=torch.float32, device=device) #pyright: ignore[reportArgumentType]

        nnetsBonds = flexprior.priors['bonds'] if 'bonds' in flexprior.priors else None
        nnetsAngles = flexprior.priors['angles'] if 'angles' in flexprior.priors else None
        nnetsDihedrals = flexprior.priors['dihedrals'] if 'dihedrals' in flexprior.priors else None

        prior_nn = ExternalNN(parameters, nnetsBonds, nnetsAngles, nnetsDihedrals, forceterms_nn, device)
        calcs += [prior_nn]

        # set the forceterms to contain only the classical terms (lj)
        forceterms = forceterms_classical
    
    # this takes the model and also constructs the classical prior
    system, forces = make_system(mols, prior_path, calcs, device, forceterms, exclusions, replicas, temperature=temperature, new_ff=not torchmd)

    # Need to fix arggggggs... - Daniel
    mock_args = ArgsMock()
    mock_args.seed = 4242                                 #pyright: ignore[reportAttributeAccessIssue]
    mock_args.timestep = timestep                         #pyright: ignore[reportAttributeAccessIssue]
    mock_args.langevin_gamma = 1                          #pyright: ignore[reportAttributeAccessIssue]
    mock_args.langevin_temperature = temperature          #pyright: ignore[reportAttributeAccessIssue]
    mock_args.minimize = None # Default from run.py       #pyright: ignore[reportAttributeAccessIssue]
    mock_args.steps = steps                               #pyright: ignore[reportAttributeAccessIssue]
    mock_args.output_period = save_steps                  #pyright: ignore[reportAttributeAccessIssue]
    mock_args.save_period = save_steps                    #pyright: ignore[reportAttributeAccessIssue]
    mock_args.replicas = replicas                         #pyright: ignore[reportAttributeAccessIssue]
    mock_args.o = output                                  #pyright: ignore[reportAttributeAccessIssue]

    dynamics(mock_args, mols[0], system, forces, device, title)

if __name__ == "__main__":
    main()
