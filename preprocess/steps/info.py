"""Write `result/info.json`, optional resume checks, touch `ok_list`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tqdm import tqdm

from ..loaders import slice_to_str

if TYPE_CHECKING:
    from ..runner import Preprocessor


def _info_dict(pre: Preprocessor) -> dict:
    tag_beta = pre.prior_builder.prior_params.get("tag_beta_turns", False)
    return {
        "input_paths": [i["path"] for i in pre.dataset_conf],
        "frame_slice": slice_to_str(pre.frame_slice),
        "pdbids": pre.trajectory.pdb_ids(),
        "optimize_forces": pre.optimize_forces,
        "box": pre.box,
        "prior_name": pre.prior_name,
        "tag_beta_turns": tag_beta,
    }


def assert_resume_compatible(pre: Preprocessor, info_dict: dict) -> None:
    if not pre.resume_preprocess:
        return
    info_path = pre.paths.info_json()
    if not info_path.exists():
        return
    with open(info_path, "rt", encoding="utf-8") as f:
        previous_info = json.load(f)
    for k in ["box", "frame_slice", "optimize_forces", "prior_name"]:
        assert info_dict[k] == previous_info[k], (
            f"Can't resume with different parameters: {k}: {info_dict[k]} != {previous_info[k]}"
        )
    if "tag_beta_turns" in previous_info:
        assert info_dict["tag_beta_turns"] == previous_info["tag_beta_turns"], (
            "Can't resume with different tag_beta_turns: "
            f"{info_dict['tag_beta_turns']} != {previous_info['tag_beta_turns']}"
        )


def write_result_metadata(pre: Preprocessor) -> None:
    pre.paths.result_dir.mkdir(parents=True, exist_ok=True)
    info_dict = _info_dict(pre)
    assert_resume_compatible(pre, info_dict)
    with open(pre.paths.info_json(), "wt", encoding="utf-8") as f:
        json.dump(info_dict, f)
    tqdm.get_lock()
    pre.paths.ok_list_txt().touch()
