
import openai
import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
load_dotenv()


class LoadConfig:
    """
    A class for loading configuration settings, including OpenAI credentials.

    This class reads configuration parameters from a YAML file and sets them as attributes.
    It also includes a method to load OpenAI API credentials.

    Attributes:
        gpt_model (str): The GPT model to be used.
        temperature (float): The temperature parameter for generating responses.
        llm_system_role (str): The system role for the language model.
        llm_function_caller_system_role (str): The system role for the function caller of the language model.

    Methods:
        __init__(): Initializes the LoadConfig instance by loading configuration from a YAML file.
        load_openai_credentials(): Loads OpenAI configuration settings.
    """

    def __init__(self) -> None:
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.gpt_model = app_config["gpt_model"]
        self.temperature = app_config["temperature"]
        self.llm_system_role = "You are a useful chatbot."
        self.llm_function_caller_system_role = app_config["llm_function_caller_system_role"]
        self.llm_system_role = app_config["llm_system_role"]

        self.load_openai_credentials()

    def load_openai_credentials(self):
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
