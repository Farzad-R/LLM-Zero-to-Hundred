
import openai
import os
from dotenv import load_dotenv

load_dotenv()


def load_cfg():
    openai.api_type = "Your API TYPE"
    openai.api_base = "Your API BASE"
    openai.api_version = "Your API VERSION"
    openai.api_key = "Your API KEY"
