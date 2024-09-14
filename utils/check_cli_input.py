import sys

from omegaconf import OmegaConf


def merge_cli_cfg(cfg: OmegaConf, cli_cfg: OmegaConf):
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

    check_key(OmegaConf.to_object(cfg), OmegaConf.to_object(cli_cfg))
    return OmegaConf.merge(cfg, cli_cfg)
