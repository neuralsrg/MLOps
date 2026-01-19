import logging
from argparse import ArgumentParser

import torch

from logq.utils.eval import evaluate
from logq.utils.dataset import get_num_items, get_test_dataloader
from logq.utils.utils import build_model, get_device, load_config


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='logq/configs/ml1m_sasrec.py')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--device', type=int, default=0)
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

test_dataloader = get_test_dataloader(config.dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length)
evaluation_result = evaluate(model, test_dataloader, config.metrics, config.recommendation_limit, 
                             config.filter_rated, device=device) 
logger.info(evaluation_result)
