import faiss
import os
import logging
import yaml
from typing import List, Dict, Optional

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.llama_cpp import LlamaCPP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def custom_chat_template(messages: List[Dict[str, str]]) -> str:
    """
    Custom chat template for LlamaCPP.

    :param messages: List of message dictionaries
    :return: Formatted prompt string
    """
    prompt=""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
        elif role == "current_user_query":
            prompt += f"Current User Query: {content}\n"
    prompt += "Assistant:"
    return prompt


class SimaticLLM:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the RAG model with the specified configuration.

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
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _initialize_model(self):
        Settings.embed_model = HuggingFaceEmbedding(self.config['embedding_model'])
        Settings.chunk_size = self.config['chunk_size']
        dim = self.config['embedding_dimension']
        self.storage_path = self.config['storage_path']
        self.faiss_index_path = os.path.join(self.storage_path, 'faiss.index')

        self.model_path = self.config['gguf_model_path']
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        logger.info(f"Model initialized with storage path: {self.storage_path}")

    def _load_documents(self, data_path: str):
        documents = SimpleDirectoryReader(data_path).load_data()
        self.index = VectorStoreIndex.from_documents(
                documents, storage_context=self.storage_context
        )
        logger.info("New index created with documents")

    def _prepare_chat_messages(self, query: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Prepare chat messages for the custom template.

        :param query: The current user query
        :param history: The conversation history
        :return: List of message dictionaries
        """
        messages = []
        for message in history:
            role = message["role"]
            messages.append({"role": role, "content": message["content"]})
        messages.append({"role": "current_user_query", "content": query})
        return messages

    def invoke(self, query: str, history: List[Dict[str, str]], data_path: Optional[str] = None, stream: Optional[bool] = False) -> str:
        """
        Invoke the RAG model with the specified query and history.
        :param stream: Whether to enable streaming output
        :param query: The user query
        :param history: Message history
        :param data_path: This is the path to the data folder for RAG
        :return:
        """

        self._load_documents(data_path)

        self.llm = LlamaCPP(
            model_path=self.model_path,
            temperature=self.config['temperature'],
            max_new_tokens=self.config['max_new_tokens'],
            context_window=self.config['context_window'],
            generate_kwargs={'top_p': self.config['top_p']},
            verbose=False
        )

        query_engine = self.index.as_query_engine(llm=self.llm, similarity_top_k=self.config['top_k'], streaming=stream)
        chat_messages = self._prepare_chat_messages(query, history)

        try:
            if stream:
                response = query_engine.query(custom_chat_template(chat_messages))
                CYAN = "\033[96m"
                RESET = "\033[0m"
                for text in response.response_gen:
                    print(f"{CYAN}{text}{RESET}", end="", flush=True)
                return None
            else:
                response = query_engine.query(custom_chat_template(chat_messages))
                return response.response
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return "I apologize, but I encountered an error while processing your query."


if __name__ == "__main__":
    try:
        stream = True
        rag = SimaticLLM()
        history = [
            {"role": "system",
             "content": "You are assisting HMI users and HRs in helping out with their tasks by providing them with "
                        "the necessary information from user manuals or resumes."
                        "Your have to respond to their queries with the context information given. Generate answers "
                        "to the 'Current User Query' alone."},
        ]
        result = rag.invoke(
            "How does Peterson's algorithm work?",
            history,
            data_path="data",
            stream=stream
        )

        if not stream:
            print(result)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
