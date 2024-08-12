import faiss
import os
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class RAG:
    def __init__(self, embedding_model="BAAI/bge-small-en-v1.5"):
        """
        Initialize the RAG model with the specified embedding model

        :param embedding_model: (optional) The name of the embedding model to use
        """
        Settings.embed_model = HuggingFaceEmbedding(embedding_model)
        Settings.llm = None
        Settings.chunk_size = 1024
        dim = 384
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def _load_documents(self, data_path):
        """
        Load the documents from the specified data path

        :param data_path: The path to the data folder
        """
        documents = SimpleDirectoryReader(data_path).load_data()
        self.index = VectorStoreIndex.from_documents(
            documents, storage_context=self.storage_context
        )
        self.index.storage_context.persist()
        self.index = load_index_from_storage(storage_context=self.storage_context)
        print("Documents have been loaded")

    def retrieve(self, data_path, query):
        """
        Retrieve the context for the specified query

        :param data_path: The path to the data folder
        :param query: The query to retrieve the context form the vector store

        :return: The context for the specified query
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist")
        self._load_documents(data_path)

        top_k = 2

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        )

        response = query_engine.query(query)

        context = ""
        for i in range(min(top_k, len(response.source_nodes))):
            context = context + response.source_nodes[i].text + "\n\n"

        prompt = f"""Query: {query}\n
        Context: {context} \n
        Response:
        """
        return prompt


if __name__ == "__main__":
    rag = RAG()
    context = rag.retrieve("/home/kausthub-kannan/Desktop/data", "What is Virtual reality?")
    print(context)
