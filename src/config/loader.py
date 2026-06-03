import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union, get_args, get_origin

import yaml
from dotenv import load_dotenv


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override values into a base configuration dictionary."""
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def dict_to_dataclass(data: Dict[str, Any], target_class) -> Any:
    """Convert nested dictionaries into the requested dataclass type when possible."""
    if not hasattr(target_class, '__dataclass_fields__'):
        return data

    kwargs = {}
    for field_name, field_type in target_class.__dataclass_fields__.items():
        if field_name in data:
            field_value = data[field_name]
            field_class = field_type.type

            if hasattr(field_class, '__dataclass_fields__'):
                kwargs[field_name] = dict_to_dataclass(field_value, field_class)
                continue

            origin = get_origin(field_class)
            args = get_args(field_class)
            if origin is list and args:
                item_class = args[0]
                if hasattr(item_class, '__dataclass_fields__') and isinstance(field_value, list):
                    kwargs[field_name] = [dict_to_dataclass(item, item_class) for item in field_value]
                else:
                    kwargs[field_name] = field_value
                continue

            kwargs[field_name] = field_value

    return target_class(**kwargs)


def load_config(
    config_name: str = 'default',
    config_dir: str = 'configs'
):
    """Load, merge, and materialize a named YAML config as a typed config object."""
    load_dotenv(override=True)

    config_path = f'{config_dir}/{config_name}.yaml'

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Configuration file not found: {config_path}')

    config_dict = load_yaml_config(config_path)

    # {./outputs/<dataset_name>/<channel>/<model_name>/<tag>}
    if config_dict['general']['work_dir'] == 'default':
        model_name = config_dict['model']['name']
        channel = f"channel{config_dict['model']['d_model']}"
        tag = config_dict['model']['tag']
        work_dir = f"{Path(config_dir).parent.absolute()}/outputs/{channel}/{model_name}/"

        work_dir += tag
        config_dict['general']['work_dir'] = work_dir
    elif config_dict['general']['work_dir'] == 'local':
        model_name = config_dict['model']['name']
        tag = config_dict['model']['tag']
        config_dict['general']['work_dir'] = f"{os.getcwd()}/{tag}"

    if 'testing_ckpt' in config_dict['general']:
        if config_dict['general']['testing_ckpt'] == 'default':
            config_dict['general']['testing_ckpt'] = f"{config_dict['general']['work_dir']}/checkpoint.pt"
        elif config_dict['general']['testing_ckpt'] == 'same':
            config_dict['general']['testing_ckpt'] = config_dict['general']['ckpt']['path']

    if not 'channel_access_token' in config_dict['linebot']:
        config_dict['linebot']['channel_access_token'] = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
    if not 'user_id' in config_dict['linebot']:
        config_dict['linebot']['user_id'] = os.getenv('LINE_USER_ID')

    # if 'Mamba2' in config_dict['model']['name']:
    #     from config.ssl.wav2vec2_mamba2 import Wav2Vec2Mamba2Config

    #     default_config = Wav2Vec2Mamba2Config()
    #     default_dict = asdict(default_config)

    #     # Backward compatibility: older YAML might put these fields under model.encoder
    #     encoder_cfg = config_dict.get('model', {}).get('encoder', {})
    #     if 'd_model' in encoder_cfg and 'd_model' not in config_dict['model']:
    #         config_dict['model']['d_model'] = encoder_cfg['d_model']
    #     if 'rms_norm_eps' in encoder_cfg and 'rms_norm_eps' not in config_dict['model']:
    #         config_dict['model']['rms_norm_eps'] = encoder_cfg['rms_norm_eps']

    #     merged_config = merge_configs(default_dict, config_dict)
    #     return dict_to_dataclass(merged_config, Wav2Vec2Mamba2Config)

    if config_name.startswith('deepfake/'):
        from config.deepfake.baseline import DeepfakeBaselineConfig, DeepfakeBaselineModelConfig

        default_config = DeepfakeBaselineConfig()
        default_dict = asdict(default_config)

        merged_config = merge_configs(default_dict, config_dict)
        return dict_to_dataclass(merged_config, DeepfakeBaselineConfig)

    if config_name.startswith('emotion/'):
        from config.emotion.baseline import EmotionBaselineConfig, EmotionBaselineModelConfig

        default_config = EmotionBaselineConfig()
        default_dict = asdict(default_config)

        merged_config = merge_configs(default_dict, config_dict)
        return dict_to_dataclass(merged_config, EmotionBaselineConfig)


def config_to_yaml(config) -> str:
    """Serialize a dataclass config object to YAML text."""
    config_dict = asdict(config)
    return yaml.dump(config_dict, default_flow_style=False, indent=2, allow_unicode=True, sort_keys=False)
