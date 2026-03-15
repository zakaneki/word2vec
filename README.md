# Word2Vec in Pure NumPy (SGNS)

A clean, dependency-minimal implementation of **Skip-Gram with Negative Sampling (SGNS)** in pure NumPy. Streams text from Hugging Face's [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset for out-of-core training on CPU, even under tight memory constraints.

## Architecture

| Module              | Responsibility                                                                                              |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| `src/word2vec.py`   | SGNS model — forward pass, loss, gradients, and vectorized SGD via `np.add.at`                              |
| `src/data_utils.py` | Vocabulary building, subsampling, context-window pairing, O(1) negative sampling (Vose's alias method)      |
| `train.py`          | Training loop with linear LR decay and streaming mini-batches                                               |
| `evaluate.py`       | Save/load model checkpoints, nearest-neighbor queries, analogy arithmetic                                   |
| `run_analogies.py`  | Batch evaluation on the standard word analogy benchmark                                                     |

### Key Design Choices

- **Both** embedding matrices (`W_t`, `W_c`) are randomly initialized for faster convergence.
- Gradients are normalized by batch size so the learning rate is batch-size-independent.
- Learning rate decays **linearly** from `lr_init` to ≈ 0 across all training tokens (matching the original word2vec C implementation).
- Negative sampling uses **Vose's Alias Method** for O(1) draws, batched per text chunk.
- Frequent-word subsampling is fully vectorized with NumPy masking.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `datasets`, `pyarrow`.

## Training

```bash
python train.py --vocab_build_limit 5000 --train_limit 20000 --epochs 3
```

| Flag                  | Default                 | Description                                                                 |
|-----------------------|-------------------------|-----------------------------------------------------------------------------|
| `--dataset_name`      | `HuggingFaceFW/fineweb` | HF dataset repository                                                       |
| `--dataset_config`    | `sample-10BT`           | HF dataset configuration                                                    |
| `--vocab_build_limit` | `5000`                  | Documents parsed to build the vocabulary (`none`/`inf` for unlimited)       |
| `--train_limit`       | `20000`                 | Documents streamed per epoch (`none`/`inf` for unlimited)                   |
| `--min_count`         | `5`                     | Words below this frequency are discarded                                    |
| `--embedding_dim`     | `100`                   | Dimensionality of word vectors                                              |
| `--window_size`       | `5`                     | Max context window (actual window is randomized per token)                  |
| `--num_negatives`     | `5`                     | Negative samples per positive pair                                          |
| `--batch_size`        | `1024`                  | Mini-batch size (512–2048 works well)                                       |
| `--epochs`            | `3`                     | Passes over the `train_limit` documents                                     |
| `--learning_rate`     | `0.025`                 | Initial LR (linearly decayed to ≈ 0)                                        |
| `--log_interval`      | `100`                   | Batches between progress logs                                               |

Checkpoints are saved to `model_checkpoints/` after training.

## Evaluation

### Nearest Neighbors (Interactive)

```python
from evaluate import load_evaluator

evaluator = load_evaluator("model_checkpoints")
print(evaluator.get_nearest_neighbors("apple", k=5))
print(evaluator.get_analogy("king", "man", "woman", k=5))
```

### Word Analogy Benchmark

Run the standard analogy test (e.g. *"king − man + woman = queen"*) against `test.txt`:

```bash
python run_analogies.py [model_dir] [test_file]
```

Both arguments are optional and default to `model_checkpoints` and `test.txt` respectively.

The test file uses the standard format:

```
: category-name
word_a word_b word_c word_d
```

Results are reported per category and overall, showing accuracy on both vocabulary-covered pairs and the full test set.

## Project Structure

```
word2vec/
├── train.py               # Training entrypoint
├── evaluate.py            # Save/load + nearest-neighbor evaluator
├── run_analogies.py       # Batch analogy benchmark
├── test.txt               # Standard analogy test set
├── requirements.txt
├── src/
│   ├── word2vec.py        # SGNS model (forward, backward, update)
│   └── data_utils.py      # Vocab, subsampling, batching, alias sampling
└── model_checkpoints/     # Saved W_t, W_c, word2id, id2word
```
