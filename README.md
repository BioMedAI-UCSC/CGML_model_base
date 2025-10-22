# CGML Base Model Repository

This repository contains the CGSchnet ML model for molecular dynamics and machine learning pipelines, providing tools for training, testing, and building upon coarse-grained neural network force fields. The model architecture is inspired by [CGSchNet](https://github.com/torchmd/torchmd-cg).

## Overview

This repository implements a machine learning framework for coarse-grained molecular dynamics simulations. The pipeline consists of three main stages:

1. **Preprocessing**: Convert all-atom MD trajectories into coarse-grained representations and fit classical prior force fields
2. **Training**: Train neural network models to predict delta forces (corrections to the prior force field)
3. **Simulation**: Run MD simulations using the trained models combined with prior force fields

The system supports multiple coarse-graining schemes (CA, CACB) and various prior force field configurations with bonds, angles, dihedrals, and non-bonded terms.

## Installation

⚠️ **Note**: Installation instructions will be provided as the code is finalized for public release.

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- Required Python packages: torchmd, moleculekit, mdtraj, numpy, pyyaml

## Model Inputs

### Training Configuration

Key hyperparameters specified in config files (e.g., `configs/config.yaml`):

- `embedding_dimension`: Dimension of learned atomic embeddings
- `num_layers`: Number of interaction layers in the neural network
- `num_rbf`: Number of radial basis functions
- `cutoff`: Interaction cutoff distance
- `max_num_neighbors`: Maximum neighbors per atom
- `activation`: Activation function (e.g., 'silu')

### Training Arguments

```bash
python train.py <input_dir> <output_dir> [options]
```

**Required:**
- `input_dir`: Path to preprocessed data directory
- `output_dir`: Directory for saving checkpoints and results

**Key Options:**
- `--config`: Path to model configuration YAML file
- `--gpus`: GPU IDs to use (e.g., "0,1,2,3" or "cpu")
- `--batch`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--wd`: Weight decay for regularization
- `--atoms-per-call`: Max atoms per sub-batch (for memory management)
- `--energy-weight`: Weight for energy loss term
- `--force-weight`: Weight for force loss term
- `--val-ratio`: Validation set ratio (default: 0.1)
- `--early-stopping`: Patience for early stopping (default: 1)

**Learning Rate Schedulers:**
- `--exp-lr`: Exponential decay (parameter: gamma)
- `--cos-lr`: Cosine annealing (parameters: T_max, eta_min)
- `--plateau-lr`: Reduce on plateau (parameters: factor, patience, threshold, min_lr)

## Preprocessing

Preprocessing converts all-atom MD trajectories into coarse-grained representations and fits prior force fields.

### Basic Usage

```bash
python preprocess.py <input_path> -o <output_dir> --prior <prior_type>
```

### Example: Preprocessing with CA Representation

```bash
python preprocess.py benchmark_set.yaml \
    -o /path/to/output/preprocessed_CA_Majewski \
    --prior CA_Majewski2022_v1 \
    --temp 300 \
    --num-cores 32
```

### Preprocessing Arguments

**Required:**
- `input`: Input directory or YAML config file listing trajectories
- `-o, --output`: Output directory for preprocessed data
- `--prior`: Prior force field type (see Available Priors below)

**Optional:**
- `--pdbids`: Specific PDB IDs to process
- `--num-frames`: Number of frames to process per trajectory
- `--frame-slice`: Python slice notation (e.g., "0:1000:10")
- `--temp`: Temperature in Kelvin (default: 300)
- `--optimize-forces`: Use statistically optimal force aggregation
- `--prior-file`: Use existing prior instead of fitting new one
- `--no-box`: Disable periodic boundary conditions
- `--num-cores`: Number of parallel processes (default: 32)

### Available Prior Force Fields

- `CA`: Basic CA representation with bonds only
- `CA_lj`: CA with bonds and repulsion
- `CA_lj_angle`: CA with bonds, angles, and repulsion
- `CA_lj_angle_dihedral`: CA with all classical terms
- `CA_lj_angleXCX_dihedralX`: CA with unified angle/dihedral parameters
- `CA_Majewski2022_v1`: Parameters from Majewski et al. 2022
- `CACB`: CA-CB representation
- `CACB_lj`: CA-CB with repulsion
- `CACB_lj_angle_dihedral`: CA-CB with all terms

### Input Data Format

Preprocessing expects HDF5 files with structure:
```
input_dir/
├── pdb_id_1/
│   └── result/
│       └── output_pdb_id_1.h5  # Contains coordinates, forces, box (optional)
├── pdb_id_2/
│   └── result/
│       └── output_pdb_id_2.h5
...
```

### Output Structure

```
output_dir/
├── priors.yaml              # Fitted prior force field parameters
├── prior_params.json        # Force field configuration
├── result/
│   ├── info.json           # Preprocessing metadata
│   └── ok_list.txt         # Successfully processed PDB IDs
└── <pdb_id>/
    ├── raw/
    │   ├── coordinates.npy  # CG coordinates
    │   ├── forces.npy       # CG forces from all-atom MD
    │   ├── deltaforces.npy  # Delta forces (MD - prior)
    │   ├── embeddings.npy   # Atom type embeddings
    │   └── box.npy         # Periodic box (if used)
    └── processed/
        └── topology.psf     # CG topology
```

## Training

Train neural network models to predict corrections to the prior force field.

### Basic Training Example

```bash
python train.py \
    /path/to/preprocessed_data \
    /path/to/results \
    --config configs/config.yaml \
    --gpus 0,1,2,3 \
    --batch 4 \
    --epochs 35 \
    --lr 0.001 \
    --exp-lr 0.85 \
    --atoms-per-call 140000 \
    --energy-weight 0.1 \
    --force-weight 1.0
```

### Training with Energy Matching

```bash
python train.py \
    /path/to/preprocessed_with_tica \
    /path/to/results_energy_matching \
    --config configs/config_cutoff2.yaml \
    --gpus 0,1,2,3 \
    --batch 4 \
    --epochs 35 \
    --lr 0.001 \
    --wd 0 \
    --exp-lr 0.85 \
    --atoms-per-call 140000 \
    --energy-weight 0.5 \
    --force-weight 1.0
```

### Resuming Training

To resume from a checkpoint, simply provide the same output directory:

```bash
python train.py \
    /path/to/preprocessed_data \
    /path/to/existing_results \
    --config configs/config.yaml \
    --gpus 0,1,2,3
```

### Training Outputs

```
results_dir/
├── checkpoint.pth           # Latest model checkpoint
├── checkpoint-best.pth      # Best validation loss checkpoint
├── history.npy             # Training/validation loss history
├── epoch_history.json      # Detailed per-epoch metrics
├── training_info.json      # Training configuration
├── priors.yaml            # Copy of prior force field
└── prior_params.json      # Copy of prior configuration
```

## Running Simulations

Use trained models to run coarse-grained MD simulations.

### Basic Simulation

```bash
python simulate.py \
    /path/to/results/checkpoint.pth \
    /path/to/starting_structure.pdb \
    --output simulation.h5 \
    --steps 1000000 \
    --save-steps 1000 \
    --temperature 300 \
    --timestep 2 \
    --replicas 1
```

### Multiple Replicas

```bash
python simulate.py \
    /path/to/results/checkpoint-best.pth \
    /path/to/structure1.pdb /path/to/structure2.pdb /path/to/structure3.pdb \
    --output replicas.pkl \
    --steps 5000000 \
    --save-steps 1000 \
    --temperature 300 \
    --replicas 3
```

### Prior-Only Simulation

To run simulations using only the classical prior (without ML corrections):

```bash
python simulate.py \
    /path/to/results/checkpoint.pth \
    /path/to/starting_structure.pdb \
    --output prior_only.h5 \
    --prior-only \
    --steps 1000000 \
    --save-steps 1000
```

### Simulation Arguments

**Required:**
- `checkpoint_path`: Path to trained model checkpoint
- `processed_path`: One or more starting structures (PDB or preprocessed directories)

**Key Options:**
- `--output`: Output trajectory file (.pdb, .h5, or .pkl)
- `--steps`: Total simulation steps
- `--save-steps`: Save frequency (frames)
- `--temperature`: Simulation temperature in Kelvin (default: 300)
- `--timestep`: Integration timestep in femtoseconds (default: 1)
- `--replicas`: Number of parallel replicas (default: 1)
- `--prior-only`: Run without ML model (classical prior only)
- `--no-box`: Disable periodic boundary conditions
- `--max-num-neighbors`: Override max neighbors parameter

### Output Formats

- **PDB format** (`.pdb`): Human-readable, compatible with visualization tools
- **HDF5 format** (`.h5`): Compact binary format with energies and metadata
- **Pickle format** (`.pkl`): Contains MDTraj trajectory objects for analysis

## Utilities

### Editing Checkpoints

Reset optimizer or scheduler state in checkpoints:

```bash
python edit_checkpoint.py /path/to/checkpoint.pth --reset-optimizer --reset-scheduler
```

View checkpoint information:

```bash
python edit_checkpoint.py /path/to/checkpoint.pth --info
```

### Learning Prior Force Fields

Train classical prior parameters directly from data:

```bash
python learn_prior.py \
    /path/to/preprocessed_data \
    /path/to/prior_output \
    --epochs 50 \
    --batch 10 \
    --lr 0.005
```

Inspired by [CGSchNet](https://github.com/torchmd/torchmd-cg).

## License

*License information will be added soon.*

## Contributing

We welcome contributions! Please use [GitHub Issues](../../issues) to:
- Request new features
- Report bugs
- Suggest improvements
- Ask questions about the pipeline

## Status

⚠️ **Note**: Code is currently being prepared for public release. Full documentation and tutorials will be available within 1-2 weeks.
