import argparse
import time
from src.data_utils import FineWebStreamer, StreamingVocabulary, SGNSBatchGenerator
from src.word2vec import Word2VecSGNS
import numpy as np


def parse_limit_arg(value: str):
    """Parse document limit where 'none'/'inf' means unlimited (-1)."""
    normalized = value.strip().lower()
    if normalized in {"none", "inf", "infinity", "all", "unlimited"}:
        return -1

    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Limit must be an integer or one of: none, inf, all"
        ) from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("Limit must be >= 0, or use 'none'/'inf'")
    return parsed

def train(args):
    """
    Main training loop for Word2Vec using NumPy and locally cached dataset.
    """
    print(f"1. Building vocabulary for limit {args.vocab_build_limit} documents...")
    streamer_vocab = FineWebStreamer(dataset_name=args.dataset_name, config_name=args.dataset_config, limit=args.vocab_build_limit)
    vocab = StreamingVocabulary(min_count=args.min_count)
    vocab.build_from_stream(streamer_vocab)
    
    print(f"Vocab size built: {len(vocab.word2id)}")
    
    # 2. Initialize Model
    print(f"2. Initializing Word2Vec model with dimension {args.embedding_dim}...")
    model = Word2VecSGNS(
        vocab_size=len(vocab.word2id), 
        embedding_dim=args.embedding_dim, 
        learning_rate=args.learning_rate
    )
    
    # Estimate batches for progress tracking
    # We estimate based on token counts collected during vocab building.
    # To keep the estimate accurate, we scale it by the ratio of train_limit vs vocab_build_limit.
    # If vocab_build_limit was unlimited, we assume train_limit is identical unless specified.
    
    tokens_kept = sum(vocab.word_counts)
    if args.vocab_build_limit > 0 and args.train_limit > 0:
        ratio = args.train_limit / args.vocab_build_limit
        tokens_kept = int(tokens_kept * ratio)
    elif args.vocab_build_limit > 0 and args.train_limit == -1:
        print("Warning: Estimating batches for unlimited training is not possible precisely; using 15M doc multiplier as rough guess.")
        ratio = 15000000 / args.vocab_build_limit
        tokens_kept = int(tokens_kept * ratio)
    
    # Average window size is window_size / 2 per side, times 2 sides.
    estimated_pairs = tokens_kept * args.window_size
    estimated_batches = estimated_pairs // args.batch_size
    total_estimated_batches = estimated_batches * args.epochs
    
    print(f"3. Beginning training... (Estimated ~{estimated_batches:,} batches per epoch)")
    batch_generator = SGNSBatchGenerator(
        vocab=vocab,
        window_size=args.window_size,
        num_negatives=args.num_negatives
    )
    
    streamer_train = FineWebStreamer(dataset_name=args.dataset_name, config_name=args.dataset_config, limit=args.train_limit)
    
    # Linear LR decay: lr decreases from lr_init to lr_min across all training.
    lr_init = args.learning_rate
    lr_min = lr_init * 0.0001  # floor so LR never hits zero
    global_batch = 0
    actual_batches_per_epoch = None  # filled in after epoch 1
    
    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        num_batches = 0
        interval_loss = 0.0
        interval_count = 0
        start_time = time.time()
        
        # We re-instantiate streamer
        if epoch > 1:
            streamer_train = FineWebStreamer(dataset_name=args.dataset_name, config_name=args.dataset_config, limit=args.train_limit)
        
        generator = batch_generator.stream_batches(streamer_train, batch_size=args.batch_size)
        
        for batch_idx, (center_ids, pos_ids, neg_ids) in enumerate(generator, start=1):
            # Linear LR decay across total training
            global_batch += 1
            progress = global_batch / max(total_estimated_batches, 1)
            model.lr = max(lr_init * (1.0 - progress), lr_min)
            
            loss = model.forward_backward_update(center_ids, pos_ids, neg_ids)
            interval_loss += loss
            interval_count += 1
            num_batches += 1
            
            if batch_idx % args.log_interval == 0:
                avg_loss = interval_loss / interval_count
                elapsed = time.time() - start_time
                pct_done = (batch_idx / estimated_batches) * 100 if estimated_batches > 0 else 0.0
                print(f"Epoch {epoch} | Batch {batch_idx:,}/{estimated_batches:,} ({pct_done:.1f}%) | "
                      f"Average Loss: {avg_loss:.4f} | LR: {model.lr:.6f} | Time: {elapsed:.2f}s")
                # Reset interval metrics
                interval_loss = 0.0
                interval_count = 0
                start_time = time.time()

        # Log any remaining interval at epoch end
        if interval_count > 0:
            avg_loss = interval_loss / interval_count
            print(f"Epoch {epoch} | Final interval | Average Loss: {avg_loss:.4f} | LR: {model.lr:.6f}")

        # After epoch 1 we know the real batch count; use it from here on
        if epoch == 1:
            actual_batches_per_epoch = num_batches
            estimated_batches = actual_batches_per_epoch
            total_estimated_batches = actual_batches_per_epoch * epochs
            print(f"(Actual batches per epoch: {actual_batches_per_epoch:,})")

        print(f"--- Epoch {epoch} Complete | Total Batches Processed: {num_batches} ---\n")

    # Save the trained model
    print("4. Saving model and vocabulary...")
    from evaluate import save_model
    save_model(model, vocab)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NumPy Word2Vec with Local HF Dataset")
    
    # Data Parameters
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb", help="HF Dataset Repository")
    parser.add_argument("--dataset_config", type=str, default="sample-10BT", help="HF Dataset Configuration")
    parser.add_argument(
        "--vocab_build_limit",
        type=parse_limit_arg,
        default=5000,
        help="Number of documents to parse to build vocabulary; use none/inf for no limit",
    )
    parser.add_argument(
        "--train_limit",
        type=parse_limit_arg,
        default=20000,
        help="Number of documents to stream per epoch; use none/inf for no limit",
    )
    parser.add_argument("--min_count", type=int, default=5, help="Minimum word frequency threshold; words appearing fewer times are discarded")
    
    # Model Hyperparameters
    parser.add_argument("--embedding_dim", type=int, default=100, help="Size of word vectors")
    parser.add_argument("--window_size", type=int, default=5, help="Context window size")
    parser.add_argument("--num_negatives", type=int, default=5, help="Number of negative samples per positive pair")
    
    # Training Loop Parameters
    parser.add_argument("--batch_size", type=int, default=1024, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of iterations over the train_limit documents")
    parser.add_argument("--learning_rate", type=float, default=0.025, help="Initial learning rate (linearly decayed to ~0)")
    parser.add_argument("--log_interval", type=int, default=100, help="Batches between print logs")
    
    args = parser.parse_args()
    train(args)
