
import yaml
import openai
from dotenv import load_dotenv
import os
from pyprojroot import here
load_dotenv()


class LoadConfig:

    def __init__(self) -> None:
        with open(here("configs/config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # memory
        self.memory_directry: str = app_config["memory"]["directory"]
        self.num_entries: int = app_config["memory"]["num_entries"]

        # llm function caller
        self.llm_function_caller_temperature: float = app_config[
            "llm_function_caller"]["temperature"]
        self.llm_function_caller_system_role: str = app_config["llm_function_caller"]["system_role"]
        self.llm_function_caller_gpt_model: str = app_config["llm_function_caller"]["gpt_model"]

        # Summarizer config
        self.llm_inference_gpt_model = app_config["llm_inference"]["gpt_model"]
        self.llm_inference_system_role = app_config[
            "llm_inference"]["system_role"]
        self.llm_inference_temperature = app_config["llm_inference"]["temperature"]

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
