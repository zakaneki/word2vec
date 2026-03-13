from evaluate import load_evaluator

evaluator = load_evaluator("model_checkpoints")

print("Nearest neighbors for 'japan':")
print(evaluator.get_nearest_neighbors("japan", k=5))

print("\nTesting Analogy: king - man + woman = ?")
print(evaluator.get_analogy("king", "man", "woman", k=5))

print("\nTesting Analogy: paris - france + italy = ?")
print(evaluator.get_analogy("paris", "france", "italy", k=5))