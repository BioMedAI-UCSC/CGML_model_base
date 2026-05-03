"""CLI: `python -m preprocess` from `base_model/` (with `module` on PYTHONPATH)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import torch
import yaml

from .config_manager import apply_yaml_defaults_to_argparser, build_preprocess_settings, load_preprocess_yaml
from .loaders import gen_input_mapping, get_prior_params_path
from .prior_builder import PRIOR_TYPES


def _peek_config_path(argv: list[str]) -> str | None:
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")

    cfg = load_preprocess_yaml(_peek_config_path(sys.argv[1:]))

    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.set_defaults(
        filter_not_processed_step_one=False,
        use_cached_fits=[],
        device_step_3="cpu",
        do_step_1=True,
        regen_cache_files=True,
        resume=False,
        prior_plots=True,
        no_box=False,
        optimize_forces=False,
        fit_min_cnt=0,
    )
    apply_yaml_defaults_to_argparser(cfg, parser)

    parser.add_argument("--config", help="YAML file with default preprocess settings (CLI overrides file)")
    parser.add_argument("input", nargs="+", help="Input directory path")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument("--pdbids", nargs="*", help="List of specific PDB IDs to process")
    parser.add_argument("--num-frames", "--num_frames", type=int, default=None, help="Number of frames to process")
    parser.add_argument("--frame-slice", type=str, default=None, help="Select frames using a python slice: start:end:stride")
    parser.add_argument("--temp", type=int, help="Temperature")
    parser.add_argument(
        "--prior",
        type=str,
        default=None,
        help="Select the prior forcefield to use, must be one of: " + ", ".join(sorted(PRIOR_TYPES.keys())),
    )
    parser.add_argument("--optimize-forces", action="store_true", help="Use statistically optimal force aggregation (Kramer 2023)")
    parser.add_argument("--prior-file", default=None, help="Use PRIOR_FILE instead of fitting a prior")
    parser.add_argument("--no-box", action="store_true", help="Don't use periodic box information")
    parser.add_argument("--prior-plots", action="store_true", help="Save prior fit plots")
    parser.add_argument("--no-prior-plots", dest="prior_plots", action="store_false", help="Do not save prior fit plots")
    parser.add_argument("--no-fit-constraints", default=False, action="store_true", help="Disable range constraints when fitting prior functions")
    parser.add_argument("--fit-min-cnt", type=int, help="Only bins with cnt > min_cnt when fitting the prior")
    parser.add_argument("--resume", action="store_true", help="Resume preprocessing")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume")
    parser.add_argument("--num-cores", type=int, help="Number of cores for parallel PDB processing")
    parser.add_argument("--jobid", type=int, default=None, help="Job id for Step 3 subset (with --totalNrJobs)")
    parser.add_argument("--totalNrJobs", type=int, default=None, help="Total array jobs for Step 3 subsetting")
    parser.add_argument("--filter-not-processed-step-one", action="store_true", help="Only PDBs with step-1 fit_ok")
    parser.add_argument("--skip-step-1", action="store_false", dest="do_step_1", help="Skip step 1 (resume steps 2–3)")
    parser.add_argument("--no-regen-cache-files", action="store_false", dest="regen_cache_files", help="Reuse prior_builder.pkl")
    parser.add_argument(
        "--device-step-3",
        type=str,
        metavar="DEV",
        default=argparse.SUPPRESS,
        help="Torch device for step 3 (e.g. cpu, cuda)",
    )
    parser.add_argument(
        "--use-cached-fits",
        nargs="*",
        default=argparse.SUPPRESS,
        metavar="TERM",
        help="Terms to load from cache in step 2 (empty list clears); omit to use YAML default",
    )

    args = parser.parse_args()
    print(args)

    output_dir = args.output
    pdbids = args.pdbids
    assert not (args.num_frames and args.frame_slice)
    if args.num_frames:
        frame_slice = slice(0, args.num_frames)
    elif args.frame_slice:
        frame_slice = slice(*[int(i) if i != "" else None for i in args.frame_slice.split(":")])
    else:
        frame_slice = slice(None)
    temp = args.temp
    optimize_forces = args.optimize_forces
    box = not args.no_box
    prior_plots = args.prior_plots
    prior_name = args.prior
    prior_file = args.prior_file
    resume_preprocess = args.resume
    num_cores = args.num_cores
    jobid = args.jobid
    total_nr_jobs = args.totalNrJobs

    if prior_file:
        assert os.path.exists(prior_file), f"Prior file does not exist: {prior_file}"
        prior_params_path = get_prior_params_path(prior_file)
        with open(prior_params_path, "r", encoding="utf-8") as f:
            prior_params = json.load(f)
            prior_configuration_name = prior_params["prior_configuration_name"]
            if prior_name is None:
                prior_name = prior_configuration_name
            elif prior_name != prior_configuration_name:
                print()
                print(
                    f'WARNING: Prior "{prior_name}" differs from the one used to build the prior file "{prior_configuration_name}"'
                )
                print()

    assert prior_name, " You must specify the prior to use with either --prior or --prior-file"

    if prior_name not in PRIOR_TYPES:
        raise RuntimeError(f"Unknown prior configuration: {prior_name}")
    print(f"Using prior: {prior_name}")
    prior_builder = PRIOR_TYPES[prior_name]()
    prior_builder.enable_fit_constraints(not args.no_fit_constraints)
    prior_builder.enable_bond_tags(False)
    prior_builder.set_min_cnt(args.fit_min_cnt)

    if "external" in prior_builder.prior_params.keys():
        mp.set_start_method("spawn")

    import matplotlib

    matplotlib.use("Agg")

    dataset_conf = []
    for i in args.input:
        if os.path.isfile(i):
            with open(args.input[0], "r") as f:
                dataset_conf += yaml.safe_load(f)
        else:
            dataset_conf += [{"path": i}]

    input_path_map = gen_input_mapping(dataset_conf)

    if pdbids:
        input_path_map = {i: input_path_map[i] for i in pdbids}

    from .runner import Preprocessor

    settings = build_preprocess_settings(args, cfg)
    preprocessor = Preprocessor(
        dataset_conf,
        input_path_map,
        output_dir,
        prior_builder,
        prior_file,
        prior_name,
        frame_slice,
        temp,
        optimize_forces,
        box,
        prior_plots,
        resume_preprocess,
        num_cores,
        jobid,
        total_nr_jobs,
        settings=settings,
    )

    preprocessor.preprocess()


if __name__ == "__main__":
    main()
