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
        self.rerank_model: str = app_config["llm_cfg"]["rerank_model"]

        # basic_rag
        self.basic_rag_index_save_dir: str = app_config["llama_index_cfg"]["basic_rag"]["index_save_dir"]

        # pagewise_rag
        self.pagewise_rag_index_save_dir: str = app_config[
            "llama_index_cfg"]["pagewise_rag"]["index_save_dir"]

        # sentence_retrieval
        self.sentence_index_save_dir: str = app_config[
            "llama_index_cfg"]["sentence_retrieval"]["index_save_dir"]
        self.sentence_window_size: int = app_config["llama_index_cfg"][
            "sentence_retrieval"]["sentence_window_size"]
        self.sentence_retrieval_similarity_top_k: int = app_config["llama_index_cfg"][
            "sentence_retrieval"]["similarity_top_k"]
        self.sentence_retrieval_rerank_top_n: int = app_config[
            "llama_index_cfg"]["sentence_retrieval"]["rerank_top_n"]

        # auto_merging_retrieval
        self.auto_merging_retrieval_index_save_dir: str = app_config["llama_index_cfg"][
            "auto_merging_retrieval"]["index_save_dir"]
        self.chunk_sizes: int = app_config["llama_index_cfg"]["auto_merging_retrieval"]["chunk_sizes"]
        self.auto_merging_retrieval_similarity_top_k: int = app_config["llama_index_cfg"][
            "auto_merging_retrieval"]["similarity_top_k"]
        self.auto_merging_retrieval_rerank_top_n: int = app_config["llama_index_cfg"][
            "auto_merging_retrieval"]["rerank_top_n"]
