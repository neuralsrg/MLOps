import csv
import logging
from argparse import ArgumentParser

import torch

from logq.utils.eval import predict
from logq.utils.dataset import get_num_items
from logq.utils.utils import build_model, get_device, load_config, csv_to_tensor_list


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='logq/configs/ml1m_sasrec.py')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--input_path', type=str, default='csv/input.csv')
parser.add_argument('--output_path', type=str, default='csv/output.csv')
parser.add_argument('--top_k', type=int, default=5)
args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

config = load_config(args.config)
num_items = get_num_items(config.dataset_name)
device = get_device(args.device)

model = build_model(config)
model = model.to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))

inputs: list[torch.Tensor] = csv_to_tensor_list(args.input_path)
outputs: list[list] = predict(model, inputs, args.top_k, device)

with open(args.output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(outputs)
