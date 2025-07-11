from memory.embedder import embed_text
from memory.vector_store import VectorStore
import numpy as np

# Initialize vector store
store = VectorStore(
    dim=384,
    index_path="data/memory.faiss",
    metadata_path="data/memory.json"
)

print(" TEACH ME FACTS (type 'done' to stop)\n")

while True:
    fact = input("→ Fact: ")
    if fact.strip().lower() == "done":
        break
    if fact.strip():
        store.add(embed_text(fact), fact)
        print("✓ Stored.\n")

print("\n ASK ME A QUESTION")
while True:
    query = input("→ Question (or 'exit'): ")
    if query.strip().lower() == "exit":
        break
    if query.strip():
        query_vec = embed_text(query)
        results = store.search(query_vec, top_k=3)

        print("Top matches:")
        for match, dist in results:
            print(f"→ {match} (distance: {round(dist, 4)})")
        print()
