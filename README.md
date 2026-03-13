# Word2Vec in Pure NumPy (SGNS)

This project is a clean, dependency-minimal implementation of the Word2Vec algorithm (specifically **Skip-gram with Negative Sampling**) in pure NumPy. It handles reading scale-aware datasets via HF's `datatrove` library for local out-of-core streaming so you can process 30GB+ of text gracefully even on CPU environments and limited memory (e.g. 16GB RAM limit constraints).

## Overview

The code comprises the standard Word2Vec mechanisms:
1. **Vocabulary Building:** Stream-processes texts to capture frequency counts and dynamically subsample highly frequent tokens (like "the").
2. **Negative Sampler:** Generates unigram occurrence tables with the standard exponent weight (0.75) distribution to pick negative context words.
3. **Mini-Batching:** Iterates across variable sliding context windows and aggregates positive/negative pairings.
4. **Vectorized Core:** The entire gradient computation and SGD parameter update is parallelized directly through NumPy arrays without heavy reliance on outer loops.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Training Loop

By default, the script streams documents from `HuggingFaceFW/fineweb` over the network chunk-by-chunk using `datatrove.pipeline.readers.ParquetReader`. 

```bash
python train.py --vocab_build_limit 5000 --train_limit 20000 --batch_size 1024 --epochs 3
```
- `--vocab_build_limit`: How many documents are parsed to construct the dataset's vocabulary constraint.
- `--train_limit`: Limit documents fetched per epoch (increase/decrease depending on how much time you have).
- `--batch_size`: Batch size for NumPy parallel updates. Mini-batches around 512-2048 are usually best.
- `--epochs`: Standard number of epochs over the `train_limit`.

## Evaluating the learned vectors

Use `evaluate.py` visually to verify the model. Check cosine-similarity using nearest neighbors via an interactive python session:

```python
from evaluate import load_evaluator

evaluator = load_evaluator("model_checkpoints")
print(evaluator.get_nearest_neighbors("apple", k=5))
```
