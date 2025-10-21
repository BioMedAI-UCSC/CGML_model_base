# CMGML Base Model Repository

This is the CGSchnet ML model repository for our molecular dynamics and machine learning pipeline, providing an ML module for training, testing, and building upon the CGSchnet model we utilize.

## Overview

This repository is part of a pipeline made of multiple components in addition to this one:

* **Regular OpenMM All Atom MD Programs and Scripts** (`openmm_generate`)  
  Standard molecular dynamics simulations using OpenMM for all-atom representations

* **WESTPA for Data Generation** (`westpa_prop`)  
  Weighted Ensemble Simulation Toolkit with Parallelization and Analysis (WESTPA) integration supporting:
  - ML-driven simulations
  - Regular MD simulations
  - Custom progress coordinates
  - Enhanced sampling workflows

* **Driver Module** (`drivers`)  
  Top Level module with ways to run the rest of the modules with pipelines linking them allowing for broader usage.

* **Benchmark Suite** (`benchmark`)  
  Qualitative and quantitative comparison tools for evaluating and validating models

* **Shared Modules** (`module`)  
  Repository containing all code and functions shared between base model, benchmark, openmm_generate, and westpa_prop.

## Status

⚠️ **IMPORTANT**: Our code is currently being ported and refactored from private repositories for public release. The full codebase with documentation and tutorials will be provided within one to two weeks.

## Contributing

We welcome contributions, feature requests, and bug reports! Please use [GitHub Issues](../../issues) to:
- Request new features
- Report bugs
- Suggest improvements
- Ask questions about the pipeline

## Installation

*Installation instructions will be added soon.*

### Prerequisites

*System requirements and dependencies will be listed here.*

### Building the Environment

*Instructions for setting up the computational environment will be added soon.*

## Getting Started

*Quick start guide will be added as modules are released.*

## Tutorials

*Step-by-step tutorials for common workflows will be provided soon.*

## Documentation

*Comprehensive documentation will be available soon.*

## License

*License information will be added soon.*

