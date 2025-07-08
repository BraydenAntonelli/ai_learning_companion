from sentence_transformers import SentenceTransformer
from typing import List

# Load a local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> List[float]:
    """
    Converts input text into an embedding vector using a local transformer model.
    """
    if not text.strip():
        raise ValueError("Cannot embed empty text")

    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
