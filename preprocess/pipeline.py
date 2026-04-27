from __future__ import annotations

from typing import TYPE_CHECKING

from .steps.info import write_result_metadata
from .steps.job_slice import apply_job_slice
from .steps.step1_pool import run_step1_parallel
from .steps.step2_pool import run_step2_parallel
from .steps.step3_pool import run_step3_parallel, write_ok_list

if TYPE_CHECKING:
    from .runner import Preprocessor


def run_preprocess_pipeline(pre: Preprocessor) -> None:
    write_result_metadata(pre)
    run_step1_parallel(pre)
    run_step2_parallel(pre)
    apply_job_slice(pre)
    run_step3_parallel(pre)
    write_ok_list(pre)
    print("Done!")
