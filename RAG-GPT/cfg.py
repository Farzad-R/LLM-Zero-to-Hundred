
import openai
import os
from dotenv import load_dotenv

load_dotenv()


def load_cfg():
    openai.api_version = os.getenv("OPENAI_API_VERSION")
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")
