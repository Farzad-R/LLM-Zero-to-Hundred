
import yaml
import openai
from dotenv import load_dotenv
import os
from pyprojroot import here
load_dotenv()


class LoadConfig:

    def __init__(self) -> None:
        with open("configs/app_config.yml") as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # llm function caller
        self.llm_function_caller_temperature: float = app_config[
            "llm_function_caller"]["temperature"]
        self.llm_function_caller_system_role: str = app_config["llm_function_caller"]["system_role"]
        self.llm_function_caller_gpt_model: str = app_config["llm_function_caller"]["gpt_model"]

        # llm summarizer
        self.llm_summarizer_temperature: float = app_config["llm_summarizer"]["temperature"]
        self.llm_summarizer_gpt_model: str = app_config["llm_summarizer"]["gpt_model"]
        self.llm_summarizer_system_role: str = app_config["llm_summarizer"]["system_role"]

        # llm rag
        self.llm_rag_temperature: float = app_config["llm_rag"]["temperature"]
        self.llm_rag_gpt_model: str = app_config["llm_rag"]["gpt_model"]
        self.llm_rag_system_role: str = app_config["llm_rag"]["system_role"]

        # memory
        self.memory_directry: str = app_config["memory"]["directory"]
        self.num_entries: int = app_config["memory"]["num_entries"]

        # RAG
        self.persist_directory: str = str(
            here(app_config["RAG"]["persist_directory"]))  # Needs to be string for the backend of Chroma
        self.k: int = app_config["RAG"]["k"]

        # Summarizer config
        self.summarizer_gpt_model = app_config["summarizer_config"]["gpt_model"]
        self.max_final_token = app_config["summarizer_config"]["max_final_token"]
        self.token_threshold = app_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role = app_config["summarizer_config"]["summarizer_llm_system_role"]
        self.character_overlap = app_config["summarizer_config"]["character_overlap"]
        self.final_summarizer_llm_system_role = app_config[
            "summarizer_config"]["final_summarizer_llm_system_role"]
        self.summarizer_temperature = app_config["summarizer_config"]["temperature"]

        # load openai credentials
        self._load_open_ai_credentials()

    def _load_open_ai_credentials(self):
        """
        Load OpenAI configuration settings.

        This function sets the OpenAI API configuration settings, including the API type, base URL,
        version, and API key. It is intended to be called at the beginning of the script or application
        to configure OpenAI settings.

        Note:
        Replace "Your API TYPE," "Your API BASE," "Your API VERSION," and "Your API KEY" with your actual
        OpenAI API credentials.
        """
        openai.api_type = os.getenv("OPENAI_API_TYPE")
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_version = os.getenv("OPENAI_API_VERSION")
        openai.api_key = os.getenv("OPENAI_API_KEY")
