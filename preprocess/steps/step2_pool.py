"""Parallel Step 2: merge per-PDB caches, fit prior, or copy fixed prior files."""

from __future__ import annotations

import os
import pickle
import shutil
from typing import TYPE_CHECKING

from tqdm import tqdm

from ..loaders import get_prior_params_path

if TYPE_CHECKING:
    from ..runner import Preprocessor


def run_step2_parallel(pre: Preprocessor) -> None:
    pdb_map = pre.trajectory.as_dict()
    if pre.prior_file:
        prior_params_path = get_prior_params_path(pre.prior_file)
        shutil.copy(pre.prior_file, pre.paths.priors_yaml())
        shutil.copy(prior_params_path, pre.paths.prior_params_json())
        return

    pb_path = pre.paths.prior_builder_pkl()
    if not pb_path.exists() or pre.settings.regen_cache_files:
        for pdbid in tqdm(pdb_map, desc="Merging cache files together"):
            pre.prior_builder.load_molecule_cache(os.fspath(pre.paths.pdb_fit(pdbid)))
        with open(pb_path, "wb") as f:
            pickle.dump(pre.prior_builder, f)
    else:
        print("Using cached prior_builder object... ")
        with open(pb_path, "rb") as f:
            pre.prior_builder = pickle.load(f)

    pre.process_step2()
    pre.prior_builder.save_prior(pre.save_path, None)
