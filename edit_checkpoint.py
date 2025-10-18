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

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("checkpoint_path", help="The model checkpoint path")
    arg_parser.add_argument("--reset-optimizer", action="store_true", help="Reset the optimizer state")
    arg_parser.add_argument("--reset-scheduler", action="store_true", help="Reset the scheduler state")
    arg_parser.add_argument("--reset-epoch", action="store_true", help="Reset the epoch and history")
    arg_parser.add_argument("--info", action="store_true", help="Show the current epoch & config")

    args = arg_parser.parse_args()

    if args.info:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        print(f"Epoch: {checkpoint['epoch']}")
        print("Config:")
        config_str = yaml.dump(checkpoint["hyper_parameters"])
        print("\n".join(["  " + i for i in config_str.split("\n")]))

    keys_to_remove = []
    if args.reset_optimizer:
        keys_to_remove.append("optimizer")
    if args.reset_scheduler:
        keys_to_remove.append("scheduler")
    if args.reset_epoch:
        keys_to_remove.append("epoch")

    if len(keys_to_remove):
        remove_checkpoint_keys(args.checkpoint_path, keys_to_remove)

