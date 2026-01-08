import os
import logging
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torchinfo import summary

import mlflow

from logq.utils.eval import evaluate
from logq.utils.utils import load_config, build_model, get_device
from logq.utils.dataset import get_train_dataloader, get_num_items, get_val_dataloader


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='logq/configs/ml1m_sasrec.py')
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# The set_experiment API creates a new experiment if it doesn't exist.
mlflow.set_experiment("SasRec training")

mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(10)

# output directory
models_dir = "models"
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

# configuration
config = load_config(args.config)
device = get_device(args.device)
num_items = get_num_items(config.dataset_name) 

# model
model = build_model(config)
optimiser = torch.optim.Adam(model.parameters())

# data
train_dataloader = get_train_dataloader(config.dataset_name, batch_size=config.train_batch_size,
                                        max_length=config.sequence_length, train_neg_per_positive=config.negs_per_pos)
val_dataloader = get_val_dataloader(config.dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length)

batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))

best_metric = float("-inf")
best_model_name = None
step = 0
steps_not_improved = 0
plot_every_num_steps = 50

model = model.to(device)
summary(model, (config.train_batch_size, config.sequence_length), batch_dim=None)

with mlflow.start_run() as run:
    mlflow.log_params({param: getattr(config, param) for param in [
        'dataset_name', 'train_batch_size', 'sequence_length', 'negs_per_pos', 
        'embedding_dim', 'num_blocks', 'num_heads', 'dropout_rate'
    ]})

    for epoch in range(config.max_epochs):
        model.train()   
        batch_iter = iter(train_dataloader)
        pbar = tqdm(range(batches_per_epoch))
        loss_sum = 0
        for batch_idx in pbar:
            step += 1
            positives, negatives = [tensor.to(device) for tensor in next(batch_iter)]
            model_input = positives[:, :-1]
            last_hidden_state, attentions = model(model_input)
            labels = positives[:, 1:]
            negatives = negatives[:, 1:, :]
            pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives], dim=-1)
            output_embeddings = model.get_output_embeddings()
            pos_neg_embeddings = output_embeddings(pos_neg_concat)
            mask = (model_input != num_items + 1).float()
            logits = torch.einsum('bse, bsne -> bsn', last_hidden_state, pos_neg_embeddings)
            gt = torch.zeros_like(logits)
            gt[:, :, 0] = 1
            
            positive_logits = logits[:, :, 0:1].to(torch.float64) #use float64 to increase numerical stability
            negative_logits = logits[:,:,1:].to(torch.float64)
            
            logits = torch.cat([positive_logits, negative_logits], -1)
            loss_per_element = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(-1)*mask
            loss = loss_per_element.sum() / mask.sum()
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            loss_sum += loss.item()
            pbar.set_description(f"Epoch {epoch} loss: {loss_sum / (batch_idx + 1):.4f}")

            if batch_idx % plot_every_num_steps == 0:
                mlflow.log_metrics(
                    {"loss": loss.item()},
                    step=epoch * batches_per_epoch + batch_idx,
                )

        mlflow.log_metrics({"loss_epoch": loss_sum / batches_per_epoch}, step=epoch + 1)
        evaluation_result = evaluate(model, val_dataloader, config.metrics, config.recommendation_limit, 
                                     config.filter_rated, device=device)
        mlflow.log_metrics({str(key).replace("@", ":"): value for key, value in evaluation_result.items()}, step=(epoch + 1) * batches_per_epoch)
        logger.info(f"Epoch {epoch} evaluation result: {evaluation_result}")
        if evaluation_result[config.val_metric] > best_metric:
            best_metric = evaluation_result[config.val_metric]
            model_name = f"models/sasrec-{config.dataset_name}-step:{step}-emb:{config.embedding_dim}-dropout:{config.dropout_rate}-metric:{best_metric}.pt" 
            logger.info(f"Saving new best model to {model_name}")
            if best_model_name is not None:
                os.remove(best_model_name)
            best_model_name = model_name
            steps_not_improved = 0
            torch.save(model.state_dict(), model_name)
        else:
            steps_not_improved += 1
            logger.warning(f"Validation metric did not improve for {steps_not_improved} steps")
            if steps_not_improved >= config.early_stopping_patience:
                logger.info(f"Stopping training, best model was saved to {best_model_name}")
                break
        
        if best_model_name is not None and os.path.exists(best_model_name):
            mlflow.log_artifact(best_model_name, artifact_path="models")
        if os.path.exists("dvc.lock"):
            mlflow.log_artifact("dvc.lock")
        if os.path.exists(args.config):
            mlflow.log_artifact(args.config, artifact_path="configs")
