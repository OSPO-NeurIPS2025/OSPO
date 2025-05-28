import os
import json
import random
import numpy as np
import torch
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

def open_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(save_root, save_name, save_file):
    save_path = os.path.join(save_root,f'{save_name}.json')
    with open(save_path, 'w') as f:
        json.dump(save_file, f, indent=4)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True) 

def get_rank(dist):
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)

    @classmethod
    def from_nested_dicts(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})

def override_from_file_name(cfg):
    if 'cfg_path' in cfg and cfg.cfg_path is not None:
        file_cfg = OmegaConf.load(cfg.cfg_path)
        cfg = OmegaConf.merge(cfg, file_cfg)
    return cfg

def override_from_cli(cfg):
    c = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, c)
    return cfg

def to_attr_dict(cfg):
    c = OmegaConf.to_container(cfg)
    cfg = AttrDict.from_nested_dicts(c)
    return cfg

def build_config(struct=False, cfg_path=None):
    if cfg_path is None:
        raise ValueError("No cfg_path given.")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, struct)
    cfg = override_from_file_name(cfg)
    cfg = override_from_cli(cfg)

    # cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg = to_attr_dict(cfg) 
    return cfg
