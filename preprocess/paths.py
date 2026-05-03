"""Filesystem layout for a single preprocess output directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessPaths:
    """All paths under the preprocess run root (`save_path` / `-o` output)."""

    root: Path

    @property
    def result_dir(self) -> Path:
        return self.root / "result"

    def info_json(self) -> Path:
        return self.result_dir / "info.json"

    def ok_list_txt(self) -> Path:
        return self.result_dir / "ok_list.txt"

    def pdb_dir(self, pdbid: str) -> Path:
        return self.root / pdbid

    def pdb_raw(self, pdbid: str) -> Path:
        return self.pdb_dir(pdbid) / "raw"

    def pdb_processed(self, pdbid: str) -> Path:
        return self.pdb_dir(pdbid) / "processed"

    def pdb_fit(self, pdbid: str) -> Path:
        return self.pdb_dir(pdbid) / "fit"

    def fit_ok(self, pdbid: str) -> Path:
        return self.pdb_fit(pdbid) / "fit_ok.txt"

    def prior_builder_pkl(self) -> Path:
        return self.root / "prior_builder.pkl"

    def prior_fit_plots_dir(self) -> Path:
        return self.root / "prior_fit_plots"

    def priors_yaml(self) -> Path:
        return self.root / "priors.yaml"

    def prior_params_json(self) -> Path:
        return self.root / "prior_params.json"

    def glob_step1_fit_ok(self):
        """Paths `root/<pdbid>/fit/fit_ok.txt` (sorted)."""
        return sorted(self.root.glob("*/fit/fit_ok.txt"))
