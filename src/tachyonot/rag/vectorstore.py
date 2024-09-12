import faiss
import numpy as np
from typing import List
from ..utils.schema import Document
from pickle import dump as p_dump, load as p_load
from ..rag.embedding import LLamaCPPEmbedding
import os


class FaissVectorStore:
    def __init__(self, dimension: int, storage_path: str):
        """
        Initialize the FaissVectorStore
        :param dimension: The dimension of the embeddings
        :param storage_path: The path to store the Faiss index and documents
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.storage_path = storage_path

    def add_documents(self, documents: List[Document]):
        """
        Add documents to the Faiss index
        :param documents: A list of Document objects to be added to the index
        """
        embeddings = np.array([doc.embedding for doc in documents]).astype("float32")
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: List[float], k: int = 2):
        """
        Search the Faiss index for the nearest neighbors of the query embedding
        :param query_embedding: The query embedding
        :param k: The number of nearest neighbors to return
        :return: A list of (Document, distance) tuples representing the nearest neighbors
        """
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)
        results = [
            (self.documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])
        ]
        return results

    def save(self):
        """
        Save the Faiss index and documents to disk
        """
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        faiss.write_index(self.index, os.path.join(self.storage_path, "faiss.index"))
        with open(os.path.join(self.storage_path, "documents.pkl"), "wb") as f:
            p_dump(self.documents, f)

    def load(self):
        """
        Load the Faiss index and documents from disk
        """
        self.index = faiss.read_index(os.path.join(self.storage_path, "faiss.index"))
        with open(os.path.join(self.storage_path, "documents.pkl"), "rb") as f:
            self.documents = p_load(f)


if __name__ == "__main__":
    store = FaissVectorStore(384, "faiss")
    root = "data"
    embeder = LLamaCPPEmbedding(
        "model/all-MiniLM-L6-v2.F32.gguf", chunk_size=100, chunk_overlap=20
    )
    documents = embeder.embed_documents(root)
    query = embeder.embed_prompt(
        "What is DROPEX and how does it help in Disaster rescue and Disaster Risk Management "
        "(DRM)?"
    )
    store.add_documents(documents)
    store.save()
    contexts = store.search(query)
    final_context = ""
    for context, distance in contexts:
        final_context += context.text + "\n"

    print(final_context)
