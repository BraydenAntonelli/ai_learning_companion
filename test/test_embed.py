from memory.embedder import embed_text
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

text1 = "The dog sat on the lawn."
text2 = "A canine was relaxing on the grass."

# Generate embeddings
vec1 = np.array(embed_text(text1))
vec2 = np.array(embed_text(text2))

# Compute cosine similarity
similarity = cosine_similarity([vec1], [vec2])[0][0]

# Print results
print(f"\nText 1: {text1}")
print(f"Text 2: {text2}")
print(f"Embedding 1 (first 5 values): {np.round(vec1[:5], 4)}")
print(f"Embedding 2 (first 5 values): {np.round(vec2[:5], 4)}")
print(f"\nCosine similarity: {similarity:.4f}")
