import numpy as np

class Word2VecSGNS:
    """
    Skip-Gram with Negative Sampling (SGNS) implemented in pure NumPy.
    Optimized for CPU mini-batch training.
    """
    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        
        # Initialize embeddings uniformly in [-scale, scale] where scale = 1.0 / embed_dim
        scale = 1.0 / embedding_dim
        
        # Target/Center Word Embeddings (W_t)
        self.W_t = np.random.uniform(-scale, scale, (vocab_size, embedding_dim))
        
        # Context Word Embeddings (W_c)
        self.W_c = np.zeros((vocab_size, embedding_dim))

    def _sigmoid(self, x):
        # Clip to prevent overflow in exp
        x = np.clip(x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-x))

    def forward_backward_update(self, center_ids, pos_ids, neg_ids):
        """
        Executes Forward pass, Loss computation, Gradient formulation, 
        and SGD Step (Backward Pass) fully vectorized per mini-batch.
        
        Args:
            center_ids: (batch_size,) target word indices
            pos_ids: (batch_size,) positive context word indices
            neg_ids: (batch_size, k) negative context word indices
            
        Returns:
            loss: scalar representing the mini-batch objective loss
        """
        batch_size = center_ids.shape[0]
        k = neg_ids.shape[1]
        
        # 1. Lookup Embeddings
        # Shapes: (batch_size, embedding_dim)
        u_w = self.W_t[center_ids]       # center word vectors
        v_c = self.W_c[pos_ids]          # positive context vectors
        
        # Shape: (batch_size, k, embedding_dim)
        v_neg = self.W_c[neg_ids]        # negative context vectors
        
        # 2. Forward Pass (Dot Products + Sigmoid)
        # Positive pair dot product
        # Element-wise multiply then sum along embedding_dim -> (batch_size,)
        score_pos = np.sum(u_w * v_c, axis=1)
        prob_pos = self._sigmoid(score_pos)  # sigma(u_w * v_c)
        
        # Negative pairs dot product
        # Broadcast u_w: (batch_size, 1, embedding_dim) 
        # Element-wise multiply with v_neg, sum along embedding_dim -> (batch_size, k)
        score_neg = np.sum(np.expand_dims(u_w, 1) * v_neg, axis=2)
        prob_neg = self._sigmoid(score_neg)  # sigma(u_w * v_neg)
        
        # 3. Compute Loss (Negative likelihood of correct classes)
        # Loss = -log(prob_pos) - sum(log(1 - prob_neg))
        # Note: 1 - sigma(x) = sigma(-x). We use 1 - prob_neg to avoid numerical instabilities if score_neg is large.
        loss_pos = -np.sum(np.log(prob_pos + 1e-10))
        loss_neg = -np.sum(np.log(1.0 - prob_neg + 1e-10))
        total_loss = (loss_pos + loss_neg) / batch_size
        
        # 4. Computed Gradients for local vectors
        # dL/dv_c = (prob_pos - 1) * u_w  [Shape: (batch_size, embedding_dim)]
        grad_v_c = np.expand_dims(prob_pos - 1.0, 1) * u_w 
        
        # dL/dv_neg = prob_neg * u_w      [Shape: (batch_size, k, embedding_dim)]
        # prob_neg: (batch_size, k) -> (batch_size, k, 1) 
        # u_w: (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)
        grad_v_neg = np.expand_dims(prob_neg, 2) * np.expand_dims(u_w, 1)
        
        # dL/du_w = (prob_pos - 1) * v_c + sum( prob_neg * v_neg ) [Shape: (batch_size, embedding_dim)]
        grad_u_w_pos = np.expand_dims(prob_pos - 1.0, 1) * v_c
        grad_u_w_neg = np.sum(np.expand_dims(prob_neg, 2) * v_neg, axis=1) # (batch_size, embedding_dim)
        grad_u_w = grad_u_w_pos + grad_u_w_neg
        
        # 5. SGD Updates (Apply Gradients locally to embedding matrices)
        # Using simple SGD loop over batch is memory efficient and handles duplicate IDs correctly
        # rather than complex scatter_add operations in pure numpy.
        
        for i in range(batch_size):
            cw = center_ids[i]
            pw = pos_ids[i]
            nws = neg_ids[i]

            # Update context embeddings
            self.W_c[pw] -= self.lr * grad_v_c[i]
            
            # Update negative context embeddings
            for j in range(k):
                self.W_c[nws[j]] -= self.lr * grad_v_neg[i, j]
                
            # Update target word embedding
            self.W_t[cw] -= self.lr * grad_u_w[i]
            
        return total_loss
