import logging
from typing import Union

import torch
from config.deepfake.baseline import DeepfakeBaselineConfig

from .deepfake import models as ModelModule


def load_model(logger: logging.Logger, cfg: DeepfakeBaselineConfig) -> torch.nn.Module:
    model_class = getattr(ModelModule, cfg.model.name)
    model: torch.nn.Module = model_class(cfg=cfg.model)
    
    logger.info(f'Load Model: \n{model}')
    
    if cfg.general.ckpt['path']:
        
        if cfg.general.ckpt['path'] == 'self':
            cfg.general.ckpt['path'] = cfg.general.work_dir + '/checkpoint.pt'
        
        model_checkpoint = torch.load(cfg.general.ckpt['path'])['model']
        logger.info(f"Try load checkpoint from {cfg.general.ckpt['path']}")
        
        modules_from_list = cfg.general.ckpt['modules']['from']
        modules_to_list = cfg.general.ckpt['modules']['to']
        
        for module_from, module_to in zip(modules_from_list, modules_to_list):
            
            if module_to == 'all':
                abs_module_to = model.state_dict().keys()
            else:
                abs_module_to = [m for m in model.state_dict().keys() if module_to in m]
                if not abs_module_to:
                    logger.warning(f"Module {module_to} not found in model state_dict")
                    continue
            
            if module_from == 'all':
                abs_module_from = model_checkpoint.keys()
            else:
                abs_module_from = [m for m in model_checkpoint.keys() if module_from in m]
                if not abs_module_from:
                    logger.warning(f"Module {module_from} not found in checkpoint state_dict")
                    continue
            
            for m_to, m_from in zip(abs_module_to, abs_module_from):
                if m_to not in model.state_dict():
                    logger.warning(f"Module {m_to} not found in model state_dict")
                    continue
                if m_from not in model_checkpoint:
                    logger.warning(f"Module {m_from} not found in checkpoint state_dict")
                    continue
                
                model.state_dict()[m_to].copy_(model_checkpoint[m_from])
            
            logger.info(f"Loaded module: {module_to} from {module_from}")
    
    model.to(cfg.general.device)
    return model