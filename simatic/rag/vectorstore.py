import faiss
import numpy as np
from typing import List
from simatic.utils.schema import Document
import pickle
from simatic.rag.embedding import LLamaCPPEmbedding
import os


class FaissVectorStore:
    def __init__(self, dimension: int, storage_path: str):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.storage_path = storage_path

    def add_documents(self, documents: List[Document]):
        embeddings = np.array([doc.embedding for doc in documents]).astype('float32')
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: List[float], k: int = 2):
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        results = [(self.documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results

    def save(self):
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        faiss.write_index(self.index, os.path.join(self.storage_path, 'faiss.index'))
        with open(os.path.join(self.storage_path, 'documents.pkl'), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self):
        self.index = faiss.read_index(os.path.join(self.storage_path, 'faiss.index'))
        with open(os.path.join(self.storage_path, 'documents.pkl'), "rb") as f:
            self.documents = pickle.load(f)


if __name__ == "__main__":
    store = FaissVectorStore(384, "faiss")
    root = "data"
    embeder = LLamaCPPEmbedding("model/all-MiniLM-L6-v2.F32.gguf", chunk_size=100, chunk_overlap=20)
    documents = embeder.embed_documents(root)
    query = embeder.embed_prompt("What is DROPEX and how does it help in Disaster rescue and Disaster Risk Management "
                                 "(DRM)?")
    store.add_documents(documents)
    store.save()
    contexts = store.search(query)
    final_context = ""
    for context, distance in contexts:
        final_context += context.text + "\n"

    print(final_context)


