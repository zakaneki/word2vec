import collections
import numpy as np
import re
from typing import List, Tuple, Generator, Iterable
from datasets import load_dataset

class SubwordTokenizer:
    """A minimal regex-based lowercasing tokenizer for Word2Vec."""
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'[a-z]+', text.lower())

class StreamingVocabulary:
    def __init__(self, min_count: int = 5, unk_token: str = "<UNK>"):
        """
        Stream over chunks of texts to build a vocabulary filtered by minimum
        frequency, and compute unigram probabilities for negative sampling.
        Words appearing fewer than min_count times are discarded.
        """
        self.min_count = min_count
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

        # Keep only words that meet the minimum frequency threshold
        top_words = [(word, count) for word, count in counter.items() if count >= self.min_count]
        top_words.sort(key=lambda x: x[1], reverse=True)
        
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

        # Build alias table for O(1) negative sampling
        self._alias_prob, self._alias_table = self._build_alias_table(self.vocab.unigram_probs)
        
    def _subsample(self, token_ids: List[int]) -> np.ndarray:
        """Randomly discard frequent words (vectorized)."""
        ids = np.asarray(token_ids, dtype=np.int32)
        keep_mask = np.random.rand(len(ids)) > self.subsample_probs[ids]
        return ids[keep_mask]

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

    @staticmethod
    def _build_alias_table(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct Vose's Alias Method tables from a probability array.

        After construction:
          - alias_prob[i]  : probability of staying at bucket i  (in [0, 1])
          - alias_table[i] : the alias index to use otherwise

        Sampling one index costs exactly two uniform draws (O(1)).
        Building the table is O(n).
        """
        n = len(probs)
        alias_prob  = np.zeros(n, dtype=np.float64)
        alias_table = np.zeros(n, dtype=np.int64)

        # Scale probabilities so they sum to n
        scaled = probs * n

        small = []
        large = []
        for i, p in enumerate(scaled):
            (small if p < 1.0 else large).append(i)

        while small and large:
            s = small.pop()
            l = large.pop()
            alias_prob[s]  = scaled[s]
            alias_table[s] = l
            scaled[l]     -= (1.0 - scaled[s])
            (small if scaled[l] < 1.0 else large).append(l)

        # Any remaining buckets are set to probability 1 (numerical stability)
        for i in large:
            alias_prob[i] = 1.0
        for i in small:
            alias_prob[i] = 1.0

        return alias_prob, alias_table

    def _negative_sample(self, k: int) -> np.ndarray:
        """Sample k negative word indices in O(k) using the pre-built alias table."""
        n = len(self._alias_prob)
        # Draw k bucket indices and k uniform values in one shot
        bucket = np.random.randint(0, n, size=k)
        coin   = np.random.random(size=k)
        # Use alias when the coin flip exceeds the bucket's own probability
        use_alias = coin >= self._alias_prob[bucket]
        bucket[use_alias] = self._alias_table[bucket[use_alias]]
        return bucket

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
            if not pairs:
                continue

            # Batch-sample all negatives for this chunk in one alias-table call
            num_pairs = len(pairs)
            neg_block = self._negative_sample(num_pairs * self.num_negatives)
            neg_block = neg_block.reshape(num_pairs, self.num_negatives)
            
            for idx, (center_id, pos_id) in enumerate(pairs):
                buffer_centers.append(center_id)
                buffer_positives.append(pos_id)
                buffer_negatives.append(neg_block[idx])
                
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
    """Yields texts from a locally cached Hugging Face dataset to prevent network drops."""
    def __init__(self, dataset_name: str = "HuggingFaceFW/fineweb", config_name: str = "sample-10BT", limit: int = -1):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.limit = limit
        
    def __iter__(self) -> Generator[str, None, None]:
        # Loads from local cache (will complete download first if needed)
        ds = load_dataset(self.dataset_name, name=self.config_name, split="train")
        for i, doc in enumerate(ds):
            if self.limit != -1 and i >= self.limit:
                break
            yield doc["text"]
