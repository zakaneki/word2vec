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
        
        # Context Word Embeddings (W_c) — random init so both matrices
        # receive non-zero gradients from the very first step.
        self.W_c = np.random.uniform(-scale, scale, (vocab_size, embedding_dim))

    def _sigmoid(self, x):
        # Clip to prevent overflow in exp (exp(-30) ≈ 9.4e-14, safely representable)
        x = np.clip(x, -30.0, 30.0)
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
        inv_batch = 1.0 / batch_size
        
        # 1. Lookup Embeddings
        # Shapes: (batch_size, embedding_dim)
        u_w = self.W_t[center_ids]       # center word vectors
        v_c = self.W_c[pos_ids]          # positive context vectors
        
        # Shape: (batch_size, k, embedding_dim)
        v_neg = self.W_c[neg_ids]        # negative context vectors
        
        # 2. Forward Pass (Dot Products + Sigmoid)
        # Positive pair: einsum 'ij,ij->i' contracts embedding_dim -> (batch_size,)
        score_pos = np.einsum('ij,ij->i', u_w, v_c)
        prob_pos = self._sigmoid(score_pos)         # sigma(u_w · v_c)
        
        # Negative pairs: einsum 'ij,ikj->ik' contracts embedding_dim -> (batch_size, k)
        score_neg = np.einsum('ij,ikj->ik', u_w, v_neg)
        prob_neg = self._sigmoid(score_neg)          # sigma(u_w · v_neg)
        
        # 3. Compute Loss (Negative log-likelihood)
        # Loss = -log(prob_pos) - sum(log(1 - prob_neg))
        loss_pos = -np.sum(np.log(prob_pos + 1e-7))
        loss_neg = -np.sum(np.log(1.0 - prob_neg + 1e-7))
        total_loss = (loss_pos + loss_neg) * inv_batch
        
        # 4. Compute Per-Sample Gradients (NOT normalized by batch_size)
        # Word2vec uses per-sample SGD: each sample contributes its full
        # gradient independently. The loss is averaged for display only.
        #
        # dL/dv_c = (prob_pos - 1) * u_w  [Shape: (batch_size, embedding_dim)]
        grad_v_c = np.expand_dims(prob_pos - 1.0, 1) * u_w 
        
        # dL/dv_neg = prob_neg * u_w      [Shape: (batch_size, k, embedding_dim)]
        # prob_neg: (batch_size, k) -> (batch_size, k, 1) 
        # u_w: (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)
        grad_v_neg = np.expand_dims(prob_neg, 2) * np.expand_dims(u_w, 1)
        
        # dL/du_w = (prob_pos - 1) * v_c + sum( prob_neg * v_neg ) [Shape: (batch_size, embedding_dim)]
        grad_u_w_pos = np.expand_dims(prob_pos - 1.0, 1) * v_c
        grad_u_w_neg = np.sum(np.expand_dims(prob_neg, 2) * v_neg, axis=1)
        grad_u_w = grad_u_w_pos + grad_u_w_neg
        
        # 5. Vectorized SGD Updates
        # np.add.at is unbuffered: duplicate indices accumulate correctly.
        np.add.at(self.W_t, center_ids, -self.lr * grad_u_w)
        np.add.at(self.W_c, pos_ids,    -self.lr * grad_v_c)
        np.add.at(self.W_c, neg_ids,    -self.lr * grad_v_neg)
            
        return total_loss
