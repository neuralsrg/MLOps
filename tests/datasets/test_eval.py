import pytest

import torch

from logq.utils.utils import load_config, dataset_available
from logq.utils.dataset import get_val_dataloader, get_test_dataloader, get_num_items, MAX_RATED_ITEMS


@pytest.fixture(params=[{'cfg_path': 'logq/configs/ml1m_sasrec.py', 'get_ds': get_val_dataloader},
                        {'cfg_path': 'logq/configs/ml1m_other.py', 'get_ds': get_val_dataloader},
                        {'cfg_path': 'logq/configs/ml1m_sasrec.py', 'get_ds': get_test_dataloader},
                        {'cfg_path': 'logq/configs/ml1m_other.py', 'get_ds': get_test_dataloader}])
def eval_dataloader(request):
    params = request.param
    config = load_config(params['cfg_path'])
    eval_dataloader = get_val_dataloader(
        dataset_name=config.dataset_name,
        batch_size=config.eval_batch_size,
        max_length=config.sequence_length
    )
    return eval_dataloader, config

@pytest.mark.skipif(
    not dataset_available("val"),
    reason="Dataset not available"
)
def test_eval_dataloader(eval_dataloader):
    eval_dataloader, config = eval_dataloader
    batch = next(iter(eval_dataloader))

    assert len(batch) == 3                                  # [input, rated, output]

    assert batch[0].shape == (config.eval_batch_size, config.sequence_length)
    assert batch[1].shape == (config.eval_batch_size, MAX_RATED_ITEMS)
    assert batch[2].shape == (config.eval_batch_size, )

    num_items = get_num_items(config.dataset_name)

    for i in range(config.eval_batch_size):
        inputs = set(batch[0][i].tolist())
        if num_items + 1 in inputs:
            inputs.remove(num_items + 1)
        rated = set(batch[1][i].tolist())
        assert inputs.issubset(rated)

    for i in range(len(batch)):
        assert batch[i].dtype == torch.int64
        assert torch.min(batch[i]).item() >= 1              # item_ids are in [1, num_items]
        assert torch.max(batch[0]).item() <= num_items + 1  # num_items + 1 is PAD
