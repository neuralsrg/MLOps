# MLOps Project

## Overwiev
Speaking of Sequential Recommendation, SasRec [1] is, perhaps, the first solution that comes to mind. SasRec is a decoder-only Transformer model that predicts the user’s next positive interaction. To date, most state-of-the-art recommender systems build on a similar idea.

Nevertheless, several studies [2, 3] argue that the loss used in the original SasRec is suboptimal for retrieval. Replacing it with a softmax loss substantially improves SasRec’s performance.

However, at web scale the softmax loss is computationally infeasible: the denominator must sum over an excessively large set of negatives. In practice, this is addressed via sampled softmax, where negatives are drawn into the denominator in-batch. This induces a popularity bias: popular items appear more frequently in the denominator and are therefore over-penalized. To debias this effect, the log‑q correction [4] has been proposed, which applies importance sampling when drawing negatives.

We identify an inaccuracy in the derivation of the conventional log‑q correction and propose a modified version [5].


## Data
For offline evaluation, we will adopt [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/).


## Experimental plan
1. Measure the performance of original SasRec.
2. Replace the loss with a sampled softmax loss and assess whether performance improves.
3. For popularity debiasing, add the log‑q correction and evaluate the resulting performance.
4. Switch to the modified log‑q correction and assess how it changes performance.


## Target metrics
Since logq-correction is only applied during training and does not affect inference latency, we measure only the model quality. Following common academic evaluation practices, our target metrics are Recall@20 and NDCG@20.


## Environment & Data 
1. Create conda environment from `environment.yml` and install our code as a package. 
```
conda env create -f environment.yml
python -m pip install -e .  # install logq as a package
```
2. For your convenience, we have already preprocessed the data, which is now stored in `./ml1m` (recommended). If you wish to download and preprocess the data manually, run
```
python src/preprocess_ml1m.py
```


## References
1. [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (2018)
2. [Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?](https://arxiv.org/abs/2309.07602) (2023)
3. [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://dl.acm.org/doi/10.1145/3366424.3386195) (2020)
4. [Sampling-bias-corrected neural modeling for large corpus item recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996) (2019)
5. [Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval](https://arxiv.org/abs/2507.09331) (2025)