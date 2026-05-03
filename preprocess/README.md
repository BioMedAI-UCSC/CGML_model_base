# Preprocess Directory

Refactored preprocessing. `base_model/preprocess.py` only forwards to `preprocess.__main__:main` (same as `python -m preprocess` from `base_model/`). The old single-file script is `preprocess_legacy.py` for reference and diffs.

## How to run

From `base_model/` so imports resolve (`preprocess` package and `module`):

```bash
python preprocess.py <inputs> -o <output_dir> --prior <PRIOR_NAME>
# same:
python -m preprocess <inputs> -o <output_dir> --prior <PRIOR_NAME>
```

Optional YAML (`--config`, e.g. `preprocess/defaults.yaml`): values are applied with `set_defaults` before `parse_args()`, so **any CLI argument you pass overrides** the corresponding YAML default (`config_manager.apply_yaml_defaults_to_argparser`):

```bash
python -m preprocess --config preprocess/defaults.yaml <inputs> -o <out> --prior CA_lj
```

`input` is one or more directories, or a YAML listing batch paths.

## How the code is organized

**Call chain:**

1. `__main__.py` — argparse, YAML (`config_manager`), inputs (`loaders`), `PriorBuilder`, then `Preprocessor(...).preprocess()`.
2. `runner.py` — `Preprocessor` class holds run state; `preprocess()` calls `pipeline.run_preprocess_pipeline(self)`.
3. `pipeline.py` — ordered stages only: `steps/info.py` → `step1_pool` → `step2_pool` → `job_slice` (optional `--jobid` / `--totalNrJobs`) → `step3_pool` (+ `write_ok_list`).
4. PDB parallelism — `step1_pool` / `step3_pool` use `worker_pool.run_pdb_pool` → `step*_threading` (pool wrapper: resume, tqdm, errors) → `process_step*` (real work). Step 3 pulls `DeltaForces` / worker counts from **`module.*`** (sources under `modules/`).

**Supporting Files**:

| Path | Role |
|------|------|
| `paths.py` | `PreprocessPaths`: canonical layout under `-o`. |
| `loaders.py` | H5 batch mapping, trajectory slices. |
| `settings.py` | `PreprocessSettings` (device step 3, resume, cached fits, …). |
| `config_manager.py` / `config_models.py` | YAML merge + CLI wiring. |
| `prior_builder.py` | `PRIOR_TYPES` registry, fit / prior I/O. |
| `trajectory_source.py` | Pluggable pdbid -> path (default H5 batch). |
| `mapping.py` | CG mapping helpers for the pipeline. |

## Design notes

- **`Preprocessor`** is the run-scoped “bag” (paths, trajectory, prior, flags, `num_cores`). **`process_step*`** does the work; **`step*_threading`** is the **multiprocessing** pool entrypoint (name is from old file, not special threading).
- **Parallelism:** outer **`mp.Pool` over PDBs**; inner **over frame/chunk workers** inside step 3 (`step3_classical_worker_count` tries to balance the two). **`multiprocessing` is not multi-node / MPI**—use disjoint jobs (e.g. `--jobid` / `--totalNrJobs` for step 3 after global steps 1–2) instead of expecting the pool to coordinate ranks.
- **`PreprocessPaths`**, **`TrajectorySource`**, YAML + CLI: one place for paths, swappable inputs, reproducible defaults.
- **Step 2** is mostly sequential merge + one global prior fit (not a per-PDB process pool); the name **`run_step2_parallel`** is legacy.
- **Tradeoffs:** pool error collection can leave partial outputs; `spawn` vs `fork` by prior type → pickling sensitivity for workers; `prior_builder.pkl` ties resume to class layout.

## Output layout

Under `-o`, paths come from `PreprocessPaths`: per-PDB `raw` / `processed` / `fit`; run-level `prior_builder.pkl`, `priors.yaml`, `prior_params.json`; `result/info.json` and `ok_list.txt`.
