# MLOps Project

## Overwiev
Speaking of Sequential Recommendation, SasRec [1] is, perhaps, the first solution that comes to mind. SasRec is a decoder-only Transformer model that predicts the user’s next positive interaction. To date, most state-of-the-art recommender systems build on a similar idea.

Nevertheless, several studies [2, 3] argue that the loss used in the original SasRec is suboptimal for retrieval. Replacing it with a softmax loss substantially improves SasRec’s performance.

However, at web scale the softmax loss is computationally infeasible: the denominator must sum over an excessively large set of negatives. In practice, this is addressed via sampled softmax, where negatives are drawn into the denominator in-batch. This induces a popularity bias: popular items appear more frequently in the denominator and are therefore over-penalized. To debias this effect, the logQ correction [4] has been proposed, which applies importance sampling when drawing negatives.

We identify an inaccuracy in the derivation of the conventional logQ correction and propose a modified version [5].


## Data
For offline evaluation, we will adopt [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/).

Dataset versioning is handled with [DVC](https://doc.dvc.org/start). [Google Drive](https://drive.google.com/drive/folders/1qQqgQGHHs_b5-ZrC2u3QAdYk28gBP7bN) is used as a remote data storage. Since August 2024 authorizing in gdrive with Python has become tricky, you have to follow this [tutorial](https://github.com/treeverse/dvc/issues/10516#issuecomment-2289652067) to pull data from Google Drive as well.

After authorizing in Google Drive, follow these steps to pull the data:
```
dvc pull
```


## Experimental plan
1. Measure the performance of original SasRec.
2. Replace the loss with a sampled softmax loss and assess whether performance improves.
3. For popularity debiasing, add the logQ correction and evaluate the resulting performance.
4. Switch to the modified logQ correction and assess how it changes performance.


## Target metrics
Since logQ correction is only applied during training and does not affect inference latency, we measure only the model quality. Following common academic evaluation practices, our target metrics are Recall@20 and NDCG@20.

Note that we measure NDCG@20 only for comparability with other works. The intended use of this retrieval model is candidate generation; hence, it's **business goal** is to retrieve as many positive candidates as possible to be ranked afterwards. From a business logic perspective, R@20 can be interpreted as a ratio of positives that the model succeeded to retrieve among top-20.


## Environment & Data 
1. Create conda environment from `environment.yml` and install our code as a package. 
```
conda env create -f environment.yml
python -m pip install -e .  # install logq as a package
```
2. For your convenience, we have already preprocessed the data, which is now stored in `./ml1m` (recommended). You can safely use it as is and skip the rest of this section. If you wish to download and preprocess the data manually (not recommended), run
```
python src/preprocess_ml1m.py
```
After that, move the ML1M dataset to `./ml1m`.


## Training models
To train original SasRec, SasRec with sampled softmax, SasRec with sampled softmax and original logQ debiasing, and SasRec with sampled softmax and proposed logQ debiasing, use `train_sasrec.py`, `train_in_batch.py`, `train_in_batch_logq_old.py`, `train_in_batch_logq_new.py`, respectively.

To run original SasRec, consider using `ml1m_sasrec.py` configuration file. For SasRec with sampled softmax, use `ml1m_other.py`. Correct configuration files are used as defaults for your convenience. 

For example, to run original SasRec on 6-th GPU, run this: 
```
python src/train_sasrec.py --device=6 --config=logq/configs/ml1m_sasrec.py  # or simply python src/train_sasrec.py --device=6
```

To run SasRec with sampled softmax and original logQ correction, run this:
```
python src/train_in_batch_logq_old.py --device=6 --config=logq/configs/ml1m_other.py  # or simply python src/train_in_batch_logq_old.py --device=6
```


## Tracking experiments 
Running training automatically creates Mlflow run. To see the results run:
```
mlflow server --port 5010
```


## Run evaluation in Docker
Build Docker image:
```
sudo docker build -t ml-app:v1 .
```

Run container:
```
sudo docker run --rm \
    -v $(pwd)/csv:/data \
    ml-app:v1 \
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


## Evaluation
To evaluate model checkpoint, run `evaluate.py` with the same configuration file used for training. For example, to evaluate SasRec with sampled softmax loss and proposed logQ correction, one would run:
```
python src/evaluate.py --config=logq/configs/ml1m_other.py --checkpoint=models/inbatch-logq-new-ml1m-step\:48-negs\:256-emb\:128-dropout\:0.5-metric\:0.023369326255676292.pt --device=6
```


## References
1. [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (2018)
2. [Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?](https://arxiv.org/abs/2309.07602) (2023)
3. [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://dl.acm.org/doi/10.1145/3366424.3386195) (2020)
4. [Sampling-bias-corrected neural modeling for large corpus item recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996) (2019)
5. [Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval](https://arxiv.org/abs/2507.09331) (2025)