
import os
import shutil
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from pyprojroot import here

load_dotenv()


class LoadConfig:
    def __init__(self) -> None:
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # LLM configs
        self.llm_engine: str = app_config["llm_config"]["engine"]
        self.llm_system_role: str = app_config["llm_config"]["llm_system_role"]
        # str() required: chromadb backend concatenates this path internally
        self.persist_directory: str = str(here(app_config["directories"]["persist_directory"]))
        self.custom_persist_directory: str = str(here(app_config["directories"]["custom_persist_directory"]))

        # OpenAI client — reads OPENAI_API_KEY from env automatically
        self.openai_client: OpenAI = OpenAI()
        self.embedding_model: OpenAIEmbeddings = OpenAIEmbeddings()

        # Retrieval configs
        self.data_directory: str = app_config["directories"]["data_directory"]
        self.k: int = app_config["retrieval_config"]["k"]
        self.embedding_model_engine: str = app_config["embedding_model_config"]["engine"]
        self.chunk_size: int = app_config["splitter_config"]["chunk_size"]
        self.chunk_overlap: int = app_config["splitter_config"]["chunk_overlap"]

        # Summarizer config
        self.max_final_token: int = app_config["summarizer_config"]["max_final_token"]
        self.token_threshold: int = app_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role: str = app_config["summarizer_config"]["summarizer_llm_system_role"]
        self.character_overlap: int = app_config["summarizer_config"]["character_overlap"]
        self.final_summarizer_llm_system_role: str = app_config["summarizer_config"]["final_summarizer_llm_system_role"]
        self.temperature: float = app_config["llm_config"]["temperature"]

        # Memory
        self.number_of_q_a_pairs: int = app_config["memory"]["number_of_q_a_pairs"]

        self.remove_directory(self.custom_persist_directory)

    def create_directory(self, directory_path: str):
        """
        Create a directory if it does not exist.

        Parameters:
            directory_path (str): The path of the directory to be created.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def remove_directory(self, directory_path: str):
        """
        Removes the specified directory.

        Parameters:
            directory_path (str): The path of the directory to be removed.

        Raises:
            OSError: If an error occurs during the directory removal process.

        Returns:
            None
        """
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(
                    f"The directory '{directory_path}' has been successfully removed.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"The directory '{directory_path}' does not exist.")
