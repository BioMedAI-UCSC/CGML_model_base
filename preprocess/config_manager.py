"""Load merged defaults from optional YAML for CLI argparse."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config_models import PreprocessYamlConfig


def load_preprocess_yaml(path: str | None) -> PreprocessYamlConfig:
    if not path:
        return PreprocessYamlConfig()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    return PreprocessYamlConfig.model_validate(raw)


def apply_yaml_defaults_to_argparser(cfg: PreprocessYamlConfig, parser) -> None:
    """Call before `parse_args()` so CLI flags override YAML."""
    s = cfg.settings
    parser.set_defaults(
        num_cores=cfg.num_cores,
        resume=cfg.resume,
        prior_plots=cfg.prior_plots,
        optimize_forces=cfg.optimize_forces,
        no_box=cfg.no_box,
        temp=cfg.temp,
        fit_min_cnt=cfg.fit_min_cnt,
        filter_not_processed_step_one=s.filter_not_processed_step_one,
        use_cached_fits=list(s.use_cached_fits),
        device_step_3=s.device_step_3,
        do_step_1=s.do_step_1,
        regen_cache_files=s.regen_cache_files,
    )


def build_preprocess_settings(args, cfg: PreprocessYamlConfig):
    from .settings import PreprocessSettings

    if hasattr(args, "use_cached_fits"):
        uc = list(args.use_cached_fits)
    else:
        uc = list(cfg.settings.use_cached_fits)
    dev = getattr(args, "device_step_3", cfg.settings.device_step_3)
    return PreprocessSettings(
        filter_not_processed_step_one=bool(args.filter_not_processed_step_one),
        use_cached_fits=uc,
        device_step_3=str(dev),
        do_step_1=bool(args.do_step_1),
        regen_cache_files=bool(args.regen_cache_files),
    )
