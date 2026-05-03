"""Pydantic schemas for optional YAML-driven preprocess defaults."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SettingsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter_not_processed_step_one: bool = False
    use_cached_fits: list[str] = Field(default_factory=list)
    device_step_3: str = "cpu"
    do_step_1: bool = True
    regen_cache_files: bool = True


class PreprocessYamlConfig(BaseModel):
    """Top-level keys in `preprocess.yaml` (unknown keys ignored)."""

    model_config = ConfigDict(extra="ignore")

    settings: SettingsSection = Field(default_factory=SettingsSection)
    num_cores: int = 32
    resume: bool = False
    prior_plots: bool = True
    optimize_forces: bool = False
    no_box: bool = False
    temp: int = 300
    fit_min_cnt: int = 0
