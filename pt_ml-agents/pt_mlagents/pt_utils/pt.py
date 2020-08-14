# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Mengqi.Ye on 8/11/2020
"""

import torch as pt
from distutils.version import LooseVersion

_is_pytorch_new = LooseVersion(pt.__version__) >= LooseVersion("1.5.0")

pt_logging = None



def set_warnings_enabled(is_enabled: bool) -> None:
    """
    Enable or disable tensorflow warnings (notably, this disables deprecation warnings.
    :param is_enabled:
    """
    level = pt_logging.WARN if is_enabled else pt_logging.ERROR
    pt_logging.set_verbosity(level)


def generate_session_config() -> pt.ConfigProto:
    """
    Generate a ConfigProto to use for ML-Agents that doesn't consume all of the GPU memory
    and allows for soft placement in the case of multi-GPU.
    """
    config = pt.ConfigProto()
    config.gpu_options.allow_growth = True
    # For multi-GPU training, set allow_soft_placement to True to allow
    # placing the operation into an alternative device automatically
    # to prevent from exceptions if the device doesn't suppport the operation
    # or the device does not exist
    config.allow_soft_placement = True
    return config
