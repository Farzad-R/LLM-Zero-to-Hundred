
import yaml
import openai
from dotenv import load_dotenv
import os
from utils.functions_prep import PrepareFunctions
from typing import List
load_dotenv()


class CFG:

    def __init__(self) -> None:
        """
        Configuration class for managing various settings and parameters in the application.

        This class initializes the configuration settings by loading them from the "configs/app_config.yml" file.
        The settings include parameters for GPT function input, GPT function caller, GPT summarizer, GPT RAG, memory,
        and OpenAI credentials.

        Attributes:
            function_json_list (List): List of functions prepared for GPT input.
            llm_function_caller_temperature (float): Temperature parameter for the GPT function caller.
            llm_function_caller_system_role (str): System role parameter for the GPT function caller.
            llm_function_caller_gpt_model (str): GPT model parameter for the GPT function caller.
            llm_summarizer_temperature (float): Temperature parameter for the GPT summarizer.
            llm_summarizer_gpt_model (str): GPT model parameter for the GPT summarizer.
            llm_summarizer_system_role (str): System role parameter for the GPT summarizer.
            llm_rag_temperature (float): Temperature parameter for the GPT RAG.
            llm_rag_gpt_model (str): GPT model parameter for the GPT RAG.
            llm_rag_system_role (str): System role parameter for the GPT RAG.
            memory_directory (str): Directory for storing memory in the application.
            num_entries (int): Number of entries to be stored in the application memory.
            persist_directory (str): Directory for persisting RAG in the application.
            k (int): Parameter k for the RAG.
        """

        with open("configs/app_config.yml") as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # prepare functions for gpt input
        self.function_json_list: List = PrepareFunctions.wrap_functions()

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
        self.persist_directory: str = app_config["RAG"]["persist_directory"]
        self.k: int = app_config["RAG"]["k"]

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
