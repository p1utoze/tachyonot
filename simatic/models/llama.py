import os
from llama_cpp import Llama
import yaml

from simatic.rag.embedding import LLamaCPPEmbedding
from simatic.utils.templates import TEMPLATE
from simatic.rag.vectorstore import FaissVectorStore

from typing import Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class SimaticLLM:
    def __init__(self, config_path: str):
        """
        Initialize the RAG model with the embedding specified configuration.

        :param config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._initialize_model()

    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from a YAML file.

        :param config_path: Path to the configuration file
        :return: Dictionary containing configuration
        """
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _initialize_model(self):
        """
        Initialize the LLM model, Faiss vector store and the embedding model.
        """
        self.llm = Llama(
            model_path=self.config["model_path"],
            temperature=self.config["temperature"],
            max_new_tokens=self.config["max_new_tokens"],
            generate_kwargs={"top_p": self.config["top_p"]},
            n_ctx=self.config["n_ctx"],
            verbose=False,
        )
        logger.info("Model initialized")

        self.vectorstore = FaissVectorStore(
            self.config["embedding_dimension"], self.config["storage_path"]
        )
        logger.info("Vector store initialized")

        if os.path.exists(self.config["storage_path"]):
            logger.info("Existing vector store found... Loading vector store")
            self.vectorstore.load()
        else:
            logger.info("No existing vector store found... Creating new vector store")

        self.embeder = LLamaCPPEmbedding(
            self.config["embedding_model"], chunk_size=100, chunk_overlap=20
        )
        logger.info("Embedder initialized")

    def store_documents(self, data_path: str):
        """
        Store the documents in the vector store.
        :param data_path: Path to the data folder
        """
        documents = self.embeder.embed_documents(data_path)
        self.vectorstore.add_documents(documents)
        logger.info("Documents stored in vector store")
        self.vectorstore.save()
        logger.info("Vector store saved")

    def invoke(self, query, stream=True):
        """
        Invoke the LLM model with the given query.
        :param query: User query
        :param stream: Enable streaming output
        :return: generator of responses
        """
        query_embedding = self.embeder.embed_prompt(query)
        results = self.vectorstore.search(query_embedding)

        final_context = ""
        for context, distance in results:
            final_context += context.text + "\n"

        prompt = TEMPLATE.format(context=final_context, question=query)

        response = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": prompt}], stream=stream
        )

        if stream:
            for message in response:
                yield message["choices"][0]["delta"].get("content", "")

        else:
            yield response["choices"][0]["delta"].get("content", "")


if __name__ == "__main__":
    chain = SimaticLLM("config.yaml")
    # chain.store_documents("data")
    generator = chain.invoke(
        "What is DROPEX and how does it help in Disaster rescue and Disaster Risk Management ("
        "DRM)?"
    )

    CYAN = "\033[96m"
    RESET = "\033[0m"
    for message in generator:
        print(f"{CYAN}{message}{RESET}", end="", flush=True)
