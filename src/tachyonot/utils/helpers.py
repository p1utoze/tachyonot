import os
import pywhispercpp.constants as constants
from pathlib import Path

def set_tiktoken_env():
    """
    Set environment variables from argparse
    """
    tiktoken_dir = Path(__file__).parents[2].absolute() / ".tiktoken_cache"
    os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_dir)
    print(os.environ["TIKTOKEN_CACHE_DIR"])

def _get_params(args) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    for arg in args.__dict__:
        if arg in constants.PARAMS_SCHEMA.keys() and getattr(args, arg) is not None:
            if constants.PARAMS_SCHEMA[arg]["type"] is bool:
                if getattr(args, arg).lower() == "false":
                    params[arg] = False
                else:
                    params[arg] = True
            else:
                params[arg] = constants.PARAMS_SCHEMA[arg]["type"](getattr(args, arg))
    return params


def get_config_path():
    try:
        config_path = os.environ["CONFIG_PATH"]
    except KeyError:
        raise ValueError("CONFIG_PATH environment variable not set")
    return config_path
