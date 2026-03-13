import numpy as np
import json
import os

class Word2VecEvaluator:
    """
    Utility class to test the trained model's nearest neighbors.
    Implements cosine similarity.
    """
    def __init__(self, W_t, word2id, id2word):
        self.W_t = W_t
        self.word2id = word2id
        self.id2word = id2word

        # Normalize rows to unit length for fast cosine similarity via dot product
        norms = np.linalg.norm(self.W_t, axis=1, keepdims=True)
        # Avoid division by zero for uninitialized vectors (e.g. UNK if never trained)
        norms[norms == 0] = 1.0 
        self.normalized_embeddings = self.W_t / norms

    def get_nearest_neighbors(self, query_word, k=5):
        if query_word not in self.word2id:
            return f"'{query_word}' is out of vocabulary."
            
        word_idx = self.word2id[query_word]
        query_vector = self.normalized_embeddings[word_idx]
        
        # Compute cosine similarity (vector dot product since both are normalizd)
        similarities = np.dot(self.normalized_embeddings, query_vector)
        
        # Find top k+1 indices (includes the query word itself)
        # argpartition is faster than argsort for large arrays
        top_indices = np.argpartition(similarities, -(k+1))[-(k+1):]
        
        # Sort the top results in descending order of similarity
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        results = []
        for idx in top_indices:
            if idx != word_idx:  # Skip the query word itself
                results.append((self.id2word[idx], similarities[idx]))
                if len(results) == k:
                    break
                    
        return results

    def get_analogy(self, word_a, word_b, word_c, k=5):
        """
        Evaluates A - B + C = ?
        e.g., "king" - "man" + "woman" = ?
        """
        for w in [word_a, word_b, word_c]:
            if w not in self.word2id:
                return f"'{w}' is out of vocabulary."
                
        idx_a = self.word2id[word_a]
        idx_b = self.word2id[word_b]
        idx_c = self.word2id[word_c]
        
        # Calculate target vector
        vec_a = self.normalized_embeddings[idx_a]
        vec_b = self.normalized_embeddings[idx_b]
        vec_c = self.normalized_embeddings[idx_c]
        
        query_vector = vec_a - vec_b + vec_c
        
        # Compute cosine similarity
        similarities = np.dot(self.normalized_embeddings, query_vector)
        
        # Retrieve top matches, excluding the input words
        top_indices = np.argpartition(similarities, -(k+3))[-(k+3):]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        results = []
        for idx in top_indices:
            if idx not in [idx_a, idx_b, idx_c]:
                results.append((self.id2word[idx], similarities[idx]))
                if len(results) == k:
                    break
                    
        return results

def save_model(model, vocab, save_dir="model_checkpoints"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    np.save(os.path.join(save_dir, "W_t.npy"), model.W_t)
    np.save(os.path.join(save_dir, "W_c.npy"), model.W_c)
    
    with open(os.path.join(save_dir, "word2id.json"), "w") as f:
        json.dump(vocab.word2id, f)
        
    with open(os.path.join(save_dir, "id2word.json"), "w") as f:
        # JSON keys must be strings, so cast integer IDs
        id2word_str = {str(k): v for k, v in vocab.id2word.items()}
        json.dump(id2word_str, f)
        
    print(f"Model and vocabulary saved to {save_dir}/")

def load_evaluator(save_dir="model_checkpoints"):
    W_t = np.load(os.path.join(save_dir, "W_t.npy"))
    
    with open(os.path.join(save_dir, "word2id.json"), "r") as f:
        word2id = json.load(f)
        
    with open(os.path.join(save_dir, "id2word.json"), "r") as f:
        id2word_raw = json.load(f)
        id2word = {int(k): v for k, v in id2word_raw.items()}
        
    return Word2VecEvaluator(W_t, word2id, id2word)
