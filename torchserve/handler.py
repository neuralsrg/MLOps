import json
from typing import List
from pathlib import Path

import torch

from ts.torch_handler.base_handler import BaseHandler

from logq.utils.eval import predict
from logq.utils.utils import build_model, get_device, load_config


class MyHandler(BaseHandler):
    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        self.device = get_device()

        self.model_names = ['sasrec', 'inbatch', 'inbatch-logq-old', 'inbatch-logq-new']
        self.models = {}

        for model_name, config in zip(
            self.model_names,
            ['ml1m_sasrec.py', 'ml1m_other.py', 'ml1m_other.py', 'ml1m_other.py']

        ):
            weights_path = Path(model_dir)/f"{model_name}-best.pt"
            config_path = Path(model_dir)/config

            config = load_config(str(config_path))
            model = build_model(config, dataset_stats_path=Path(model_dir)/"dataset_stats.json").to(self.device)
            model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            model.eval()

            self.models[model_name] = model

        self.initialized = True

    def preprocess(self, data):
        if not data or "body" not in data[0]:
            return None
        
        body = data[0]["body"]

        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")

        history: List[int] = body["history"]
        top_k: int = body["top_k"]
        model_name: str = body["model_name"]

        return history, top_k, model_name

    def inference(self, inputs):
        history, top_k, model_name = inputs
        history = [torch.tensor(history, dtype=torch.long)]
        assert model_name in self.model_names
        outputs = predict(self.models[model_name], history, top_k, self.device)
        return outputs[0]

    def postprocess(self, output):
        return [{'output': output}]
