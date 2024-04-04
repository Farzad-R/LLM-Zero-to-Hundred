
import yaml
from langchain.embeddings.openai import OpenAIEmbeddings
from pyprojroot import here
from utils.app_utils import Apputils


class LoadRAGConfig:
    """
    Load configuration parameters for RAG (Retrieval-Augmented Generation) system from a YAML file.

    This class loads configuration parameters from the 'configs/rag_gpt.yml' file and provides
    attributes for accessing these parameters.

    Attributes:
        llm_engine (str): The engine parameter for the LLM (Long Language Model).
        llm_system_role (str): The system role parameter for the LLM.
        persist_directory (str): The persist directory parameter for RAG configuration.
        custom_persist_directory (str): The custom persist directory parameter for RAG configuration.
        embedding_model (OpenAIEmbeddings): The embedding model used in RAG.
        data_directory (str): The data directory parameter for RAG configuration.
        k (int): The k parameter for retrieval configuration (number of retrieved documents).
        embedding_model_engine (str): The engine parameter for the embedding model configuration.
        chunk_size (int): The chunk size parameter for splitter configuration.
        chunk_overlap (int): The chunk overlap parameter for splitter configuration.
        max_final_token (int): The maximum final token parameter for summarizer configuration.
        token_threshold (float): The token threshold parameter for summarizer configuration.
        summarizer_llm_system_role (str): The summarizer LLM system role parameter for summarizer configuration.
        character_overlap (int): The character overlap parameter for summarizer configuration.
        final_summarizer_llm_system_role (str): The final summarizer LLM system role parameter for summarizer configuration.
        temperature (float): The temperature parameter for LLM configuration.
        fetch_k (int): The number of documents to fetch from the vector database.
        lambda_param (float): The lambda parameter for MMR (Maximal Marginal Relevance).
        number_of_q_a_pairs (int): The number of Q&A pairs parameter for memory configuration.
        rag_reference_server_url (str): The RAG reference server URL parameter for serving configuration.

    """

    def __init__(self) -> None:
        with open(here("configs/rag_gpt.yml")) as cfg:
            rag_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # LLM configs
        self.llm_engine = rag_config["llm_config"]["engine"]
        self.llm_system_role = rag_config["llm_config"]["llm_system_role"]
        # here() throws error for wsl2
        self.persist_directory = str(
            rag_config["directories"]["persist_directory"])  # needs to be strin for summation in chromadb backend: self._settings.require("persist_directory") + "/chroma.sqlite3"
        self.custom_persist_directory = str(
            rag_config["directories"]["custom_persist_directory"])
        self.embedding_model = OpenAIEmbeddings()

        # Retrieval configs
        self.data_directory = rag_config["directories"]["data_directory"]
        self.k = rag_config["retrieval_config"]["k"]
        self.embedding_model_engine = rag_config["embedding_model_config"]["engine"]
        self.chunk_size = rag_config["splitter_config"]["chunk_size"]
        self.chunk_overlap = rag_config["splitter_config"]["chunk_overlap"]

        # Summarizer config
        self.max_final_token = rag_config["summarizer_config"]["max_final_token"]
        self.token_threshold = rag_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role = rag_config["summarizer_config"]["summarizer_llm_system_role"]
        self.character_overlap = rag_config["summarizer_config"]["character_overlap"]
        self.final_summarizer_llm_system_role = rag_config[
            "summarizer_config"]["final_summarizer_llm_system_role"]
        self.temperature = rag_config["llm_config"]["temperature"]

        self.fetch_k = rag_config["mmr_search_config"]["fetch_k"]
        self.lambda_param = rag_config["mmr_search_config"]["lambda_param"]

        # Memory
        self.number_of_q_a_pairs = rag_config["memory"]["number_of_q_a_pairs"]

        # clean up the upload doc vectordb if it exists
        Apputils.create_directory(self.persist_directory)
        Apputils.remove_directory(self.custom_persist_directory)
