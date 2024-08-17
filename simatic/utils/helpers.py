import os

import pywhispercpp.utils as utils
import pywhispercpp.constants as constants


def _get_params(args) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    for arg in args.__dict__:
        if arg in constants.PARAMS_SCHEMA.keys() and getattr(args, arg) is not None:
            if constants.PARAMS_SCHEMA[arg]['type'] is bool:
                if getattr(args, arg).lower() == 'false':
                    params[arg] = False
                else:
                    params[arg] = True
            else:
                params[arg] = constants.PARAMS_SCHEMA[arg]['type'](getattr(args, arg))
    return params


def get_config_path():
    try:
        config_path = os.environ['CONFIG_PATH']
    except KeyError:
        raise ValueError("CONFIG_PATH environment variable not set")
    return config_path
