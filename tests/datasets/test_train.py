import pytest

import torch

from logq.utils.utils import load_config, dataset_available
from logq.utils.dataset import get_train_dataloader, get_num_items


@pytest.fixture(params=[{'cfg_path': 'logq/configs/ml1m_sasrec.py'},
                        {'cfg_path': 'logq/configs/ml1m_other.py'}])
def train_dataloader(request):
    params = request.param
    config = load_config(params['cfg_path'])
    train_dataloader = get_train_dataloader(
        dataset_name=config.dataset_name,
        batch_size=config.train_batch_size,
        max_length=config.sequence_length,
        train_neg_per_positive=config.negs_per_pos
    )
    return train_dataloader, config

@pytest.mark.skipif(
    not dataset_available("train"),
    reason="Dataset not available"
)
def test_train_dataloader(train_dataloader):
    train_dataloader, config = train_dataloader
    batch = next(iter(train_dataloader))

    assert len(batch) == 2

    assert batch[0].shape == (config.train_batch_size, config.sequence_length + 1)
    assert batch[1].shape == (config.train_batch_size, config.sequence_length + 1, config.negs_per_pos)

    assert batch[0].dtype == torch.int64
    assert batch[1].dtype == torch.int64

    assert torch.min(batch[0]).item() >= 1              # item_ids are in [1, num_items]
    assert torch.min(batch[1]).item() >= 1              # item_ids are in [1, num_items]

    num_items = get_num_items(config.dataset_name)
    assert torch.max(batch[0]).item() <= num_items + 1  # num_items + 1 is PAD
    assert torch.max(batch[1]).item() <= num_items + 1  # num_items + 1 is PAD
