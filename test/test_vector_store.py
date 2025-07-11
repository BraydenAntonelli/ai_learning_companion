from memory.embedder import embed_text
from memory.vector_store import VectorStore
import numpy as np

# Set up the vector store (same dimensions as MiniLM: 384)
store = VectorStore(
    dim=384,
    index_path="data/memory.faiss",
    metadata_path="data/memory.json"
)

# Teaching the AI two facts
sentences = [
    "The dog sat on the lawn.",
    "A feline was sleeping on the sofa."
]

for sentence in sentences:
    store.add(embed_text(sentence), sentence)

# Ask a similar question
query = "A cat was napping indoors."
query_vector = embed_text(query)
results = store.search(query_vector, top_k=2)

# Print results
print(f"\nQuery: {query}")
print("Top matches:")
for match, score in results:
    print(f"→ {match} (distance: {round(score, 4)})")
