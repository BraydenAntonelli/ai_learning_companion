from memory.embedder import embed_text
import numpy as np

test_text = "Reinforcement learning uses rewards and penalties."

try:
    vector = embed_text(test_text)
    print(f"Embedding length: {len(vector)}")
    print(f"First 5 values: {np.round(vector[:5], 4)}")
except Exception as e:
    print("Something went wrong:", e)
# This code tests the embed_text function from the memory.embedder module.