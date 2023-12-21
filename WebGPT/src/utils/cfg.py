
import openai
import os
from dotenv import load_dotenv

load_dotenv()


def load_cfg():
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
