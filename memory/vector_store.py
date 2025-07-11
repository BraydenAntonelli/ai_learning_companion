import faiss
import numpy as np
import os
import json
from typing import List, Tuple

class VectorStore:
    def __init__(self, dim: int, index_path: str, metadata_path: str):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        index_exists = os.path.exists(index_path)
        metadata_exists = os.path.exists(metadata_path)

        if index_exists and metadata_exists:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []


    def add(self, embedding: List[float], text: str):
        vec = np.array(embedding, dtype="float32").reshape(1, -1)
        self.index.add(vec)
        self.metadata.append(text)
        self._save()

    def search(self, embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        vec = np.array(embedding, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], dist))
        return results

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
