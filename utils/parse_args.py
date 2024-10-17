import os
import sys
from argparse import ArgumentParser

from omegaconf import OmegaConf


def check_key(cfg_dict: dict, cli_cfg_dict: dict, prefix=""):
    for key in cli_cfg_dict.keys():
        if key not in cfg_dict:
            param = f"{prefix}.{key}" if prefix else key
            sys.exit(f"config param `{param}` isn't valid")
        if isinstance(cli_cfg_dict[key], dict):
            check_key(
                cfg_dict[key],
                cli_cfg_dict[key],
                f"{prefix}.{key}" if prefix else key,
            )


def parse_args(method: str):
    # parse arguments for configuration files
    parser = ArgumentParser()
    parser.add_argument("--dataset-cfg", type=str, default="./configs/dataset/voc.yaml")
    parser.add_argument("--io-cfg", type=str, default="./configs/io/io.yaml")
    parser.add_argument(
        "--method-cfg", type=str, default=f"./configs/method/{method}.yaml"
    )
    args, unknown = parser.parse_known_args()
    dataset_cfg = OmegaConf.load(args.dataset_cfg)
    io_cfg = OmegaConf.load(args.io_cfg)
    method_cfg = OmegaConf.load(args.method_cfg)

    # parse arguments for command line configuration
    cli_args = []
    for arg in unknown:
        if "=" in arg:
            cli_args.append(arg)
        else:
            cli_args[-1] += f" {arg}"
    cli_cfg = OmegaConf.from_dotlist(cli_args)

    # merge all configurations
    config = OmegaConf.merge(dataset_cfg, io_cfg, method_cfg)
    check_key(OmegaConf.to_object(config), OmegaConf.to_object(cli_cfg))
    config = OmegaConf.merge(config, cli_cfg)
    config.output_path = config.output_path[config.dataset]
    os.makedirs(config.output_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_path, f"{method}.yaml"))

    return config
