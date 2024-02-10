import yaml
from pyprojroot import here
import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file


class LoadConfig:

    def __init__(self) -> None:
        with open(here("configs/config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        self.documents_dir: str = app_config["documents_dir"]
        self.gpt_model: str = app_config["llm_cfg"]["gpt_model"]
        self.embed_model_name: str = app_config["llm_cfg"]["embed_model_name"]
        self.temperature: str = app_config["llm_cfg"]["temperature"]

        # Langchain rag: (chunk and overlap)
        self.splitter_type: str = app_config["langchain_cfg"]["splitter_type"]
        self.token_vector_db_save_dir: str = app_config["langchain_cfg"]["token_vector_db_save_dir"]
        self.recursive_vector_db_save_dir: str = app_config[
            "langchain_cfg"]["recursive_vector_db_save_dir"]
        self.langchain_recursive_chunk_size: str = app_config["langchain_cfg"]["recursive_chunk_size"]
        self.langchain_recursive_chunk_overlap: str = app_config[
            "langchain_cfg"]["recursive_chunk_overlap"]
        self.langchain_token_chunk_size: str = app_config["langchain_cfg"]["token_chunk_size"]
        self.langchain_token_chunk_overlap: str = app_config["langchain_cfg"]["token_chunk_overlap"]
        self.langchain_k: str = app_config["langchain_cfg"]["k"]
