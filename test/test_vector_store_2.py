from memory.embedder import embed_text
from memory.vector_store import VectorStore
import numpy as np

# Initialize vector store
store = VectorStore(
    dim=384,
    index_path="data/memory.faiss",
    metadata_path="data/memory.json"
)

# Clear memory for clean test (use only for isolated testing)
store.index.reset()
store.metadata = []

# Teaching facts
sentences = [
    "The dog sat on the lawn.",
    "A feline was sleeping on the sofa.",
    "Reinforcement learning uses rewards and penalties.",
    "Photosynthesis is the process plants use to convert sunlight into energy.",
    "The Battle of Hastings occurred in 1066.",
    "E = mc^2 is Einstein's theory of mass-energy equivalence."
]

for sentence in sentences:
    store.add(embed_text(sentence), sentence)

# Test queries
queries = [
    "How do plants get energy from the sun?",
    "What happened in 1066?",
    "How do machines learn from rewards?",
    "Tell me something about dogs.",
    "Explain Einstein's equation about mass."
]

# Run search
for query in queries:
    query_vec = embed_text(query)
    results = store.search(query_vec, top_k=2)

    print(f"\nQuery: {query}")
    print("Top matches:")
    for match, score in results:
        print(f"→ {match} (distance: {round(score, 4)})")
