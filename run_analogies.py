import sys
import numpy as np
import os
import time
from evaluate import load_evaluator

def run_analogy_test(model_dir="model_checkpoints", test_file="test.txt"):
    """
    Evaluates the Word2Vec model on the standard analogy task.
    Format of test.txt: A B C D (meaning A is to B as C is to D)
    Formula: vector(D_pred) = vector(B) - vector(A) + vector(C)
    """
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        return
        
    if not os.path.exists(test_file):
        print(f"Error: Test file '{test_file}' not found.")
        return
        
    print(f"Loading evaluator from {model_dir}...")
    start_time = time.time()
    t_start = start_time
    evaluator = load_evaluator(model_dir)
    print(f"Loaded in {time.time() - start_time:.2f}s. Vocab size: {len(evaluator.word2id)}")
    
    word2id = evaluator.word2id
    id2word = evaluator.id2word
    embeddings = evaluator.normalized_embeddings
    
    categories = {}
    current_category = None
    
    # Pre-lowercase all vocabulary words for robust matching
    lower_word2id = {str(word).lower(): idx for word, idx in word2id.items()}
    lower_id2word = {idx: str(word).lower() for word, idx in word2id.items()}

    print(f"Running analogies from {test_file}...")
    start_time = time.time()

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            if line.startswith(':'):
                current_category = line[1:].strip()
                categories[current_category] = {'correct': 0, 'total': 0, 'seen': 0}
            else:
                if current_category is None:
                    continue
                
                words = line.lower().split()
                if len(words) != 4:
                    continue
                
                categories[current_category]['total'] += 1
                a, b, c, expected = words
                
                # Check if all words exist in the model's vocabulary
                if a not in lower_word2id or b not in lower_word2id or c not in lower_word2id or expected not in lower_word2id:
                    continue
                    
                categories[current_category]['seen'] += 1
                
                idx_a = lower_word2id[a]
                idx_b = lower_word2id[b]
                idx_c = lower_word2id[c]
                
                # Equation: B - A + C
                target_vec = embeddings[idx_b] - embeddings[idx_a] + embeddings[idx_c]
                
                # Compute cosines against all vocab
                sims = np.dot(embeddings, target_vec)
                
                # Exclude the query words from being selected as the answer
                sims[idx_a] = -np.inf
                sims[idx_b] = -np.inf
                sims[idx_c] = -np.inf
                
                best_idx = np.argmax(sims)
                
                if lower_id2word[best_idx] == expected:
                    categories[current_category]['correct'] += 1

    total_correct = 0
    total_seen = 0
    total_all = 0
    
    print("\n--- Analogy Test Results ---")
    for cat, stats in categories.items():
        corr = stats['correct']
        seen = stats['seen']
        tot = stats['total']
        
        acc_seen = (corr / seen * 100) if seen > 0 else 0
        acc_tot = (corr / tot * 100) if tot > 0 else 0
        
        print(f"Category '{cat}':")
        print(f"   Accuracy (on seen words): {acc_seen:5.2f}% ({corr}/{seen})")
        print(f"   Accuracy (overall):       {acc_tot:5.2f}% ({corr}/{tot})")
        
        total_correct += corr
        total_seen += seen
        total_all += tot
        
    print("-" * 30)
    overall_seen = (total_correct / total_seen * 100) if total_seen > 0 else 0
    overall_all = (total_correct / total_all * 100) if total_all > 0 else 0
    print("OVERALL SUMMARY:")
    print(f"Total processing time: {time.time() - start_time:.2f}s")
    print(f"Tested on seen vocabulary pairs: {total_seen} / {total_all} total questions")
    print(f"Overall Accuracy (on seen  pairs): {overall_seen:5.2f}% ({total_correct}/{total_seen})")
    print(f"Overall Accuracy (on total pairs): {overall_all:5.2f}% ({total_correct}/{total_all})")

if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'model_checkpoints'
    test_path = sys.argv[2] if len(sys.argv) > 2 else 'test.txt'
    run_analogy_test(model_path, test_path)
