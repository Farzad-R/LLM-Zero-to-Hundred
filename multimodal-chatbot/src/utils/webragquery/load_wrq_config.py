
import yaml
from pyprojroot import here
from utils.app_utils import Apputils


class LoadWRQConfig:
    """
    Load configuration parameters for WebRAGQuery system from a YAML file.

    This class loads configuration parameters from the 'configs/webragquery.yml' file and provides
    attributes for accessing these parameters.

    Attributes:
        llm_function_caller_temperature (float): The temperature parameter for the LLM function caller model.
        llm_function_caller_system_role (str): The system role parameter for the LLM function caller model.
        llm_function_caller_gpt_model (str): The GPT model parameter for the LLM function caller model.
        llm_summarizer_temperature (float): The temperature parameter for the LLM summarizer model.
        llm_summarizer_gpt_model (str): The GPT model parameter for the LLM summarizer model.
        llm_summarizer_system_role (str): The system role parameter for the LLM summarizer model.
        llm_rag_temperature (float): The temperature parameter for the LLM RAG model.
        llm_rag_gpt_model (str): The GPT model parameter for the LLM RAG model.
        llm_rag_system_role (str): The system role parameter for the LLM RAG model.
        persist_directory (str): The persist directory parameter for RAG configuration.
        k (int): The k parameter (number of retrieved content) for RAG configuration.
        summarizer_gpt_model (str): The GPT model parameter for summarizer configuration.
        max_final_token (int): The maximum final token parameter for summarizer configuration.
        token_threshold (float): The token threshold parameter for summarizer configuration.
        summarizer_llm_system_role (str): The summarizer LLM system role parameter for summarizer configuration.
        character_overlap (int): The character overlap parameter for summarizer configuration.
        final_summarizer_llm_system_role (str): The final summarizer LLM system role parameter for summarizer configuration.
        summarizer_temperature (float): The temperature parameter for summarizer configuration.
    """

    def __init__(self) -> None:
        with open("configs/webragquery.yml") as cfg:
            wrq_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # llm function caller
        self.llm_function_caller_temperature: float = wrq_config[
            "llm_function_caller"]["temperature"]
        self.llm_function_caller_system_role: str = wrq_config["llm_function_caller"]["system_role"]
        self.llm_function_caller_gpt_model: str = wrq_config["llm_function_caller"]["gpt_model"]

        # llm summarizer
        self.llm_summarizer_temperature: float = wrq_config["llm_summarizer"]["temperature"]
        self.llm_summarizer_gpt_model: str = wrq_config["llm_summarizer"]["gpt_model"]
        self.llm_summarizer_system_role: str = wrq_config["llm_summarizer"]["system_role"]

        # llm rag
        self.llm_rag_temperature: float = wrq_config["llm_rag"]["temperature"]
        self.llm_rag_gpt_model: str = wrq_config["llm_rag"]["gpt_model"]
        self.llm_rag_system_role: str = wrq_config["llm_rag"]["system_role"]

        # RAG
        self.persist_directory: str = str(
            here(wrq_config["RAG"]["persist_directory"]))  # Needs to be string for the backend of Chroma
        self.k: int = wrq_config["RAG"]["k"]

        # Summarizer config
        self.summarizer_gpt_model = wrq_config["summarizer_config"]["gpt_model"]
        self.max_final_token = wrq_config["summarizer_config"]["max_final_token"]
        self.token_threshold = wrq_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role = wrq_config["summarizer_config"]["summarizer_llm_system_role"]
        self.character_overlap = wrq_config["summarizer_config"]["character_overlap"]
        self.final_summarizer_llm_system_role = wrq_config[
            "summarizer_config"]["final_summarizer_llm_system_role"]
        self.summarizer_temperature = wrq_config["summarizer_config"]["temperature"]

        self.fetch_k = wrq_config["mmr_search_config"]["fetch_k"]
        self.lambda_param = wrq_config["mmr_search_config"]["lambda_param"]
        Apputils.remove_directory(self.persist_directory)
