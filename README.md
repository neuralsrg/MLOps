# MLOps Project

## Overwiev
Speaking of Sequential Recommendation, SasRec [1] is, perhaps, the first solution that comes to mind. SasRec is a decoder-only Transformer model that predicts the user’s next positive interaction. To date, most state-of-the-art recommender systems build on a similar idea.

Nevertheless, several studies [2, 3] argue that the loss used in the original SasRec is suboptimal for retrieval. Replacing it with a softmax loss substantially improves SasRec’s performance.

However, at web scale the softmax loss is computationally infeasible: the denominator must sum over an excessively large set of negatives. In practice, this is addressed via sampled softmax, where negatives are drawn into the denominator in-batch. This induces a popularity bias: popular items appear more frequently in the denominator and are therefore over-penalized. To debias this effect, the logQ correction [4] has been proposed, which applies importance sampling when drawing negatives.

We identify an inaccuracy in the derivation of the conventional logQ correction and propose a modified version [5].


## Experimental plan
1. Measure the performance of original SasRec.
2. Replace the loss with a sampled softmax loss and assess whether performance improves.
3. For popularity debiasing, add the logQ correction and evaluate the resulting performance.
4. Switch to the modified logQ correction and assess how it changes performance.


## Target metrics
Since logQ correction is only applied during training and does not affect inference latency, we measure only the model quality. Following common academic evaluation practices, our target metrics are Recall@20 and NDCG@20.

Note that we measure NDCG@20 only for comparability with other works. The intended use of this retrieval model is candidate generation; hence, it's **business goal** is to retrieve as many positive candidates as possible to be ranked afterwards. From a business logic perspective, R@20 can be interpreted as a ratio of positives that the model succeeded to retrieve among top-20.


## Environment
1. Create conda environment from `environment.yml` and install our code as a package. 
```
conda env create -f environment.yml
python -m pip install -e .  # install logq as a package
```


## Data
For offline evaluation, we will adopt [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/).

Dataset versioning is handled with [DVC](https://doc.dvc.org/start). [Google Drive](https://drive.google.com/drive/folders/1qQqgQGHHs_b5-ZrC2u3QAdYk28gBP7bN) is used as a remote data storage. Since August 2024 authorizing in gdrive with Python has become tricky, you have to follow this [tutorial](https://github.com/treeverse/dvc/issues/10516#issuecomment-2289652067) to pull data from Google Drive as well as to push it.

Following the instruction above, obtain `gdrive_client_id` and `gdrive_client_secret` and store them in `.dvc/config.local`:
```
['remote "storage"']
    gdrive_client_id = <YOUR CLIENT ID>
    gdrive_client_secret = <YOUR SECRET>
```

After authorizing in Google Drive, pull the data:
```
dvc pull
```

To ensure data and checkpoints are up to date, run:
```
dvc repro
```


## Training models
To train original SasRec, SasRec with sampled softmax, SasRec with sampled softmax and original logQ debiasing, and SasRec with sampled softmax and proposed logQ debiasing, use `train_sasrec.py`, `train_in_batch.py`, `train_in_batch_logq_old.py`, `train_in_batch_logq_new.py`, respectively.

To run original SasRec, consider using `ml1m_sasrec.py` configuration file. For SasRec with sampled softmax, use `ml1m_other.py`. Correct configuration files are used as defaults for your convenience. 

For example, to run original SasRec on 6-th GPU, run this: 
```
python src/train_sasrec.py \
    --device=6
    --config=logq/configs/ml1m_sasrec.py
# or simply python src/train_sasrec.py --device=6
```

To run SasRec with sampled softmax and original logQ correction, run this:
```
python src/train_in_batch_logq_old.py \
    --device=6 \
    --config=logq/configs/ml1m_other.py
# or simply python src/train_in_batch_logq_old.py --device=6
```


## Tracking experiments 
Running training automatically creates Mlflow run. To see the results run:
```
mlflow server --port 5010
```


## Evaluation
To evaluate model checkpoint, run `evaluate.py` with the same configuration file used for training. For example, to evaluate SasRec with sampled softmax loss and proposed logQ correction, one would run:
```
python src/evaluate.py \
    --config=logq/configs/ml1m_other.py \
    --checkpoint=models/inbatch-logq-new-best.pt \
    --device=6
```


## Run evaluation in Docker
Build Docker image:
```
docker build -t ml-app:v1 .
```

Or pull it from [Dockerhub](https://hub.docker.com/r/neuralsrg/ml-app):
```
docker pull neuralsrg/ml-app:v1
```

Run container:
```
docker run --rm \
    -v $(pwd)/csv:/data \
    ml-app:v1 \  # OR neuralsrg/ml-app:v1
    --config=logq/configs/ml1m_other.py \
    --checkpoint=models/inbatch-logq-new-best.pt \
    --input_path=/data/input.csv \
    --output_path=/data/output.csv \
    --top_k=5
```

Check predictions:
```
head csv/output.csv
```


## Torchserve inference 

Archive inputs:
```
mkdir torchserve/model-store && \
torch-model-archiver \
    --model-name archive \
    --version 1.0 \
    --serialized-file models/sasrec-best.pt \
    --handler torchserve/handler.py \
    --extra-files "logq/configs/ml1m_sasrec.py,logq/configs/ml1m_other.py,ml1m/dataset_stats.json,ml1m/item_cnt.pkl,models/inbatch-best.pt,models/inbatch-logq-old-best.pt,models/inbatch-logq-new-best.pt" \
    --export-path torchserve/model-store \
    --force
```

Build docker image:
```
docker build -f torchserve/Dockerfile -t mymodel-serve:v1 .
```

Or pull it from [Dockerhub](https://hub.docker.com/r/neuralsrg/torchserve):
```
docker pull neuralsrg/torchserve:v1
```

Run docker container:
```
docker run -d -p 8070:8080 -p 8071:8081 mymodel-serve:v1  # or neuralsrg/torchserve:v1
```

Send POST request and get output:
```
curl -X \
    POST http://localhost:8070/predictions/model \
    -H "Content-Type: application/json" \
    --data-binary @json/input.json
```

To stop running container:
```
docker stop <CONTAINER ID>
```


## References
1. [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (2018)
2. [Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?](https://arxiv.org/abs/2309.07602) (2023)
3. [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://dl.acm.org/doi/10.1145/3366424.3386195) (2020)
4. [Sampling-bias-corrected neural modeling for large corpus item recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996) (2019)
5. [Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval](https://arxiv.org/abs/2507.09331) (2025)