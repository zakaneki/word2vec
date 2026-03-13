import argparse
import time
from src.data_utils import FineWebStreamer, StreamingVocabulary, SGNSBatchGenerator
from src.word2vec import Word2VecSGNS
import numpy as np

def train(args):
    """
    Main training loop for Word2Vec using NumPy and datatrove streamer.
    Handles data streaming, vocabulary building, and model SGD iterations.
    """
    print(f"1. Building vocabulary for limit {args.vocab_build_limit} documents...")
    streamer_vocab = FineWebStreamer(start_url=args.dataset_url, limit=args.vocab_build_limit)
    vocab = StreamingVocabulary(vocab_size=args.max_vocab)
    vocab.build_from_stream(streamer_vocab)
    
    print(f"Vocab size built: {len(vocab.word2id)}")
    
    # 2. Initialize Model
    print(f"2. Initializing Word2Vec model with dimension {args.embedding_dim}...")
    model = Word2VecSGNS(
        vocab_size=len(vocab.word2id), 
        embedding_dim=args.embedding_dim, 
        learning_rate=args.learning_rate
    )
    
    # Training Loop Over Stream
    print(f"3. Beginning training over {args.train_limit} streamed documents...")
    batch_generator = SGNSBatchGenerator(
        vocab=vocab,
        window_size=args.window_size,
        num_negatives=args.num_negatives
    )
    
    streamer_train = FineWebStreamer(start_url=args.dataset_url, limit=args.train_limit)
    
    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        # We re-instantiate streamer since datatrove ParquetReader consumes the generator
        if epoch > 1:
            streamer_train = FineWebStreamer(start_url=args.dataset_url, limit=args.train_limit)
        
        generator = batch_generator.stream_batches(streamer_train, batch_size=args.batch_size)
        
        for batch_idx, (center_ids, pos_ids, neg_ids) in enumerate(generator, start=1):
            loss = model.forward_backward_update(center_ids, pos_ids, neg_ids)
            total_loss += loss
            num_batches += 1
            
            if batch_idx % args.log_interval == 0:
                avg_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} | Batch {batch_idx} | Average Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
                # Reset metrics for next interval
                total_loss = 0.0
                start_time = time.time()
                
        print(f"--- Epoch {epoch} Complete | Total Batches Processed: {num_batches} ---")
        
        # Basic learning rate decay
        model.lr *= args.lr_decay
        print(f"Decayed Learning Rate to {model.lr:.6f}\n")

    # Save the trained model
    print("4. Saving model and vocabulary...")
    from evaluate import save_model
    save_model(model, vocab)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NumPy Word2Vec with Datatrove FineWeb Stream")
    
    # Data Streaming Parameters
    parser.add_argument("--dataset_url", type=str, default="hf://datasets/HuggingFaceFW/fineweb/sample/10BT", help="HF Dataset URL")
    parser.add_argument("--vocab_build_limit", type=int, default=5000, help="Number of documents to parse to build vocabulary (saves RAM)")
    parser.add_argument("--train_limit", type=int, default=20000, help="Number of documents to stream per epoch")
    parser.add_argument("--max_vocab", type=int, default=20000, help="Max vocabulary size")
    
    # Model Hyperparameters
    parser.add_argument("--embedding_dim", type=int, default=100, help="Size of word vectors")
    parser.add_argument("--window_size", type=int, default=5, help="Context window size")
    parser.add_argument("--num_negatives", type=int, default=5, help="Number of negative samples per positive pair")
    
    # Training Loop Parameters
    parser.add_argument("--batch_size", type=int, default=1024, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of iterations over the train_limit documents")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Multiplicative learning rate decay per epoch")
    parser.add_argument("--log_interval", type=int, default=100, help="Batches between print logs")
    
    args = parser.parse_args()
    train(args)
