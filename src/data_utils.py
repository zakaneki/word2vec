import collections
import numpy as np
import re
from typing import List, Tuple, Generator, Iterable

# Lazy load datatrove only when initialized to decouple environments
try:
    from datatrove.pipeline.readers import ParquetReader
except ImportError:
    pass

class SubwordTokenizer:
    """A minimal regex-based lowercasing tokenizer for Word2Vec."""
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'[a-z]+', text.lower())

class StreamingVocabulary:
    def __init__(self, vocab_size: int, unk_token: str = "<UNK>"):
        """
        Stream over chunks of texts to build a capped vocabulary and 
        compute unigram probabilities for negative sampling.
        """
        self.max_vocab_size = vocab_size
        self.unk_token = unk_token
        
        self.word2id = {}
        self.id2word = {}
        self.word_counts = [] # List of tuples (word, count)
        self.total_tokens = 0
        self.unigram_probs = None
        self.sampling_table = None

    def build_from_stream(self, text_stream: Iterable[str]):
        """Builds frequency mappings from an iterator of texts."""
        counter = collections.Counter()
        
        for text in text_stream:
            tokens = SubwordTokenizer.tokenize(text)
            self.total_tokens += len(tokens)
            counter.update(tokens)

        # Keep top (vocab_size - 1) words to leave room for UNK
        top_words = counter.most_common(self.max_vocab_size - 1)
        
        # Insert UNK first
        self.word2id[self.unk_token] = 0
        self.id2word[0] = self.unk_token
        
        # Build mapping and frequencies
        freqs = [0] # index 0 is for UNK, frequency handled below
        unknown_count = self.total_tokens - sum(count for _, count in top_words)
        freqs[0] = unknown_count

        for idx, (word, count) in enumerate(top_words, start=1):
            self.word2id[word] = idx
            self.id2word[idx] = word
            freqs.append(count)

        self.word_counts = freqs
        self._build_unigram_table()

    def _build_unigram_table(self):
        """Constructs unigram distribution ^ 0.75 for negative sampling."""
        freqs = np.array(self.word_counts, dtype=np.float64)
        pow_freqs = np.power(freqs, 0.75)
        self.unigram_probs = pow_freqs / np.sum(pow_freqs)
        
    def encode(self, text: str) -> List[int]:
        tokens = SubwordTokenizer.tokenize(text)
        return [self.word2id.get(token, 0) for token in tokens]

class SGNSBatchGenerator:
    """
    Generates training data containing target word, positive context,
    and multiple negative samples via streaming chunks.
    """
    def __init__(self, 
                 vocab: StreamingVocabulary, 
                 window_size: int = 5, 
                 num_negatives: int = 5,
                 subsample_threshold: float = 1e-4):
        self.vocab = vocab
        self.window_size = window_size
        self.num_negatives = num_negatives
        
        # Subsampling probabilities (Mikolov et al., 2013) -> P(w_i) = 1 - sqrt(t / f(w_i))
        # Total words count
        total_words = sum(self.vocab.word_counts)
        # Avoid division by zero by setting 0 frequencies to a small number
        freqs = np.array(self.vocab.word_counts) / total_words
        freqs[freqs == 0] = 1e-10
        # Prob of deleting a word
        self.subsample_probs = 1.0 - np.sqrt(subsample_threshold / freqs)
        self.subsample_probs = np.clip(self.subsample_probs, 0.0, 1.0)
        
    def _subsample(self, token_ids: List[int]) -> List[int]:
        """Randomly discard frequent words."""
        filtered = []
        for tid in token_ids:
            if np.random.rand() > self.subsample_probs[tid]:
                filtered.append(tid)
        return filtered

    def _generate_pairs(self, token_ids: List[int]) -> List[Tuple[int, int]]:
        """Standard window slicing for center and positive pairs."""
        pairs = []
        # Dynamic window size (random between 1 and max_window_size)
        for i, center_id in enumerate(token_ids):
            reduced_window = np.random.randint(1, self.window_size + 1)
            start = max(0, i - reduced_window)
            end = min(len(token_ids), i + reduced_window + 1)
            
            for j in range(start, end):
                if i != j:
                    pairs.append((center_id, token_ids[j]))
        return pairs

    def _negative_sample(self, k: int) -> np.ndarray:
        """Sample negative word indices according to probability distribution."""
        # Note: Using unigram_probs to weight selection directly in choice is fine, 
        # normally alias method is faster but np.random.choice is okay with small vocab on CPU.
        return np.random.choice(len(self.vocab.id2word), size=k, p=self.vocab.unigram_probs)

    def stream_batches(self, text_stream: Iterable[str], batch_size: int = 256) -> Generator:
        """Yields mini-batches of (center_ids, pos_ids, neg_ids)."""
        buffer_centers = []
        buffer_positives = []
        buffer_negatives = []
        
        for text in text_stream:
            token_ids = self.vocab.encode(text)
            token_ids = self._subsample(token_ids)
            
            if len(token_ids) < 2:
                continue
                
            pairs = self._generate_pairs(token_ids)
            
            for center_id, pos_id in pairs:
                buffer_centers.append(center_id)
                buffer_positives.append(pos_id)
                buffer_negatives.append(self._negative_sample(self.num_negatives))
                
                if len(buffer_centers) >= batch_size:
                    yield (
                        np.array(buffer_centers, dtype=np.int32),
                        np.array(buffer_positives, dtype=np.int32),
                        np.stack(buffer_negatives).astype(np.int32)
                    )
                    buffer_centers = []
                    buffer_positives = []
                    buffer_negatives = []
                    
        # Yield remainder
        if len(buffer_centers) > 0:
            yield (
                np.array(buffer_centers, dtype=np.int32),
                np.array(buffer_positives, dtype=np.int32),
                np.stack(buffer_negatives).astype(np.int32)
            )

class FineWebStreamer:
    """Wraps Datatrove ParquetReader to stream texts to limit RAM usage."""
    def __init__(self, start_url: str = "hf://datasets/HuggingFaceFW/fineweb/sample/10BT", limit: int = None):
        if 'ParquetReader' not in globals():
            raise ImportError("Please install datatrove: pip install datatrove")
        self.url = start_url
        self.limit = limit
        
    def __iter__(self) -> Generator[str, None, None]:
        # Using limit here because reading all 30GB to RAM is dangerous
        # We will stream documents one by one and yield their text fields.
        reader = ParquetReader(self.url, limit=self.limit)
        for doc in reader():
            yield doc.text
