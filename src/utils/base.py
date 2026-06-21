import inspect
import logging
import os
import random

import numpy as np
import torch
from config import BaseConfig


def setup_freeze(cfg: BaseConfig, logger: logging.Logger, model: torch.nn.Module) -> None:
    """Apply configured parameter freezing and unfreezing before training."""
    if cfg.general.freeze != []:
        update_parameter_requires_grad(logger, model, 'freezing', cfg.general.freeze, False)
    if cfg.general.unfreeze != []:
        update_parameter_requires_grad(logger, model, 'unfreezing', cfg.general.unfreeze, True)

def setup_tf32(cfg: BaseConfig, logger: logging.Logger) -> None:
    """Configure TF32 settings for supported CUDA devices."""
    if cfg.general.device != 'cuda':
        return

    if not torch.cuda.is_available():
        logger.info('CUDA is not available, using default precision')
        return

    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    major, minor = torch.cuda.get_device_capability(current_device)

    logger.info(f'Using GPU: {gpu_name}')
    logger.info(f'GPU compute capability: {major}.{minor}')

    # TF32 is available on NVIDIA Ampere and later, i.e. compute capability >= 8.0.
    if major >= 8:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info('TF32 enabled for better performance on supported GPUs')
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        logger.info('TF32 not supported on this GPU, using default precision')

def set_seed(seed=39, deterministic=False):
    """Seed Python, NumPy, and PyTorch RNGs and configure cuDNN determinism."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def filter_params_for_class(cls, params: dict) -> dict:
    """Keep only non-empty parameters accepted by a class constructor."""
    sig = inspect.signature(cls.__init__)
    valid_params = {}
    for k, v in params.items():
        # is None or empty string
        if v is None:
            continue
        if isinstance(v, str) and v == '':
            continue
        if k in sig.parameters:
            valid_params[k] = v
    return valid_params

def update_parameter_requires_grad(
        logger: logging.Logger,
        model: torch.nn.Module,
        action: str,
        param_names: list,
        requires_grad: bool
    ) -> None:
    """Update requires_grad for named model parameters matching configured patterns."""
    logger.info(f'{action.capitalize()} model parameters...')
    if param_names == ['all']:
        param_names = [name for name, _ in model.named_parameters()]
    for name, param in model.named_parameters():
        if any(target in name for target in param_names):
            param.requires_grad = requires_grad

    logger.info(f'{action.capitalize()} model {param_names} done.')
