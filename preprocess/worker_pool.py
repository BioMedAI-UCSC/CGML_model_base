"""Multiprocessing pool helpers for per-PDB work (Steps 1 and 3)."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, Mapping

from tqdm import tqdm


def process_init(counter: Any) -> None:
    """Assign stable worker names for tqdm positioning."""
    with counter.get_lock():
        idx = int(counter.value)
        counter.value += 1
    mp.current_process().name = f"PreprocessWorker-{idx}"


def run_pdb_pool(
    num_cores: int,
    pdbids: Mapping[str, Any],
    task_fn: Callable[[str], None],
    desc: str,
) -> Dict[str, str]:
    """Run task_fn(pdbid) in parallel; return pdbid -> error message for failures."""
    error_list: Dict[str, str] = {}
    n = len(pdbids)
    thread_counter = mp.Value("i", 0, lock=True)
    tqdm.get_lock()
    with tqdm(total=n, desc=desc, dynamic_ncols=True) as pbar:
        with mp.Pool(num_cores, initializer=process_init, initargs=(thread_counter,)) as pool:
            pending: Dict[Any, str] = {}
            for pdbid in pdbids:
                pending[pool.apply_async(task_fn, args=(pdbid,))] = pdbid
            while pending:
                for result in list(pending.keys()):
                    if result.ready():
                        try:
                            result.get()
                        except Exception as e:
                            error_list[pending[result]] = str(e)
                        finally:
                            del pending[result]
                pbar.n = n - len(pending)
                pbar.refresh()
                if pending:
                    next(iter(pending)).wait(1)
    return error_list
