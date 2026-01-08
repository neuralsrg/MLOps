import importlib

import torch

from .gsasrec import GSASRec
from .dataset import get_num_items
from ..configs.config import GSASRecExperimentConfig


def load_config(config_file: str) -> GSASRecExperimentConfig:
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def build_model(config: GSASRecExperimentConfig):
    num_items = get_num_items(config.dataset_name) 
    model = GSASRec(num_items, sequence_length=config.sequence_length, embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads, num_blocks=config.num_blocks, dropout_rate=config.dropout_rate)
    return model

def get_device(gpu_id: int = 0):
    device = "cpu"
    if torch.cuda.is_available():
        device=f"cuda:{gpu_id}"
    return device

def csv_to_tensor_list(file_path):
    tensor_list = []
    with open(file_path, 'r') as file:
        for line in file:
            integers = [int(x.strip()) for x in line.strip().split(',') if x.strip()]
            tensor_list.append(torch.tensor(integers, dtype=torch.long))
    return tensor_list
