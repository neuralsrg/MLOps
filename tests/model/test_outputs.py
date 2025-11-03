import pytest

import torch

from logq.utils.eval import evaluate
from logq.utils.utils import load_config, build_model, get_device
from logq.utils.dataset import get_test_dataloader, get_num_items


@pytest.fixture(params=[{'cfg_path': 'logq/configs/ml1m_sasrec.py'},
                        {'cfg_path': 'logq/configs/ml1m_other.py'}])
def config(request):
    params = request.param
    config = load_config(params['cfg_path'])
    return config

def test_model_outputs(config):
    device = get_device()
    num_items = get_num_items(config.dataset_name)
    test_dataloader = get_test_dataloader(
        dataset_name=config.dataset_name,
        batch_size=config.eval_batch_size,
        max_length=config.sequence_length
    )
    model = build_model(config).to(device)
    data, rated, _ = next(iter(test_dataloader))
    data, rated = data.to(device), rated.to(device)

    for r in [rated, None]:
        items, scores = model.get_predictions(data, config.recommendation_limit, r)
        assert items.dtype == torch.int64

        assert items.shape == scores.shape == (config.eval_batch_size, config.recommendation_limit)
        assert torch.min(items).item() >= 0
        assert torch.max(items).item() <= num_items

        sorted_scores, _ = torch.sort(scores, dim=1, descending=True)
        assert torch.equal(scores, sorted_scores)

        if r is not None:
            for user in range(config.eval_batch_size):
                recommended = set(items[user].tolist())
                not_to_recommend = set(rated[user].tolist())
                assert recommended.intersection(not_to_recommend) == set()

    evaluation_result = evaluate(
        model=model,
        data_loader=test_dataloader,
        metrics=config.metrics,
        limit=config.recommendation_limit,
        filter_rated=config.filter_rated,
        device=device
    )
    for key, value in evaluation_result.items():
        key = str(key)
        if key.startswith('R@'):
            assert 0 <= value <= 1
            continue
        if key.startswith('nDCG@'):
            assert 0 <= value <= 1
            continue
        raise AssertionError(f'Unknown metric: {key}')
