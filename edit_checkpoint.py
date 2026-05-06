#!/usr/bin/env python3

import os
import torch
import datetime
import shutil
import yaml

def remove_checkpoint_keys(checkpoint_path, keys):
    """Remove the given keys from a model checkpoint"""

    # Make a backup of the checkpoint befor modifying it
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pth")
    assert os.path.isfile(checkpoint_path)

    backup_path, backup_ext = os.path.splitext(checkpoint_path)
    backup_path = backup_path + "-backup-" + datetime.datetime.now().isoformat() + backup_ext

    assert not os.path.exists(backup_path)

    shutil.copy(checkpoint_path, backup_path)

    checkpoint_dict = torch.load(checkpoint_path)

    print("Checkpoint (initial) keys:", checkpoint_dict.keys())

    for k in keys:
        if k in checkpoint_dict:
            del checkpoint_dict[k]

    print("Checkpoint (final) keys:", checkpoint_dict.keys())

    torch.save(checkpoint_dict, checkpoint_path)

# ---------------------------------------------------------------------------
# Entry-point helpers: collection → validation → processing
# ---------------------------------------------------------------------------

def build_parser():
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Inspect or modify a model checkpoint file.")
    arg_parser.add_argument("checkpoint_path", help="Path to the model checkpoint (.pth or directory)")
    arg_parser.add_argument("--reset-optimizer", action="store_true", help="Remove the optimizer state")
    arg_parser.add_argument("--reset-scheduler", action="store_true", help="Remove the scheduler state")
    arg_parser.add_argument("--reset-epoch", action="store_true", help="Remove the epoch counter and history")
    arg_parser.add_argument("--info", action="store_true", help="Print the current epoch and model config")
    return arg_parser


def validate_config(cfg) -> None:
    path = cfg.checkpoint_path
    is_valid = os.path.isfile(path) or (os.path.isdir(path) and os.path.isfile(os.path.join(path, "checkpoint.pth")))
    assert is_valid, f"Checkpoint not found: {path}"


def process_config(cfg) -> list:
    keys_to_remove = []
    if cfg.reset_optimizer:
        keys_to_remove.append("optimizer")
    if cfg.reset_scheduler:
        keys_to_remove.append("scheduler")
    if cfg.reset_epoch:
        keys_to_remove.append("epoch")
    return keys_to_remove


if __name__ == "__main__":
    from module.base_config import BaseConfig
    cfg = BaseConfig(build_parser())
    validate_config(cfg)

    if cfg.info:
        checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
        print(f"Epoch: {checkpoint['epoch']}")
        print("Config:")
        config_str = yaml.dump(checkpoint["hyper_parameters"])
        print("\n".join(["  " + i for i in config_str.split("\n")]))

    keys_to_remove = process_config(cfg)
    if keys_to_remove:
        remove_checkpoint_keys(cfg.checkpoint_path, keys_to_remove)

