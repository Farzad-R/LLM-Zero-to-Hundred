
import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
import shutil

load_dotenv()


class LoadConfig:
    """
    A class for loading configuration settings and managing directories.

    This class loads various configuration settings from the 'app_config.yml' file,
    including language model (LLM) configurations, retrieval configurations, summarizer
    configurations, and memory configurations. It also sets up OpenAI API credentials
    and performs directory-related operations such as creating and removing directories.
    """

    def __init__(self) -> None:
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # LLM configs
        self.llm_engine = app_config["llm_config"]["engine"]
        self.llm_system_role_with_history = app_config["llm_config"]["llm_system_role_with_history"]
        self.llm_system_role_without_history = app_config[
            "llm_config"]["llm_system_role_without_history"]
        self.persist_directory = str(here(
            app_config["directories"]["persist_directory"]))  # needs to be strin for summation in chromadb backend: self._settings.require("persist_directory") + "/chroma.sqlite3"
        self.custom_persist_directory = str(here(
            app_config["directories"]["custom_persist_directory"]))
        self.gemma_token = os.getenv("GEMMA_TOKEN")
        self.device = app_config["llm_config"]["device"]
        # Retrieval configs
        self.data_directory = app_config["directories"]["data_directory"]
        self.k = app_config["retrieval_config"]["k"]
        self.chunk_size = int(app_config["splitter_config"]["chunk_size"])
        self.chunk_overlap = int(
            app_config["splitter_config"]["chunk_overlap"])
        self.temperature = float(app_config["llm_config"]["temperature"])
        self.add_history = bool(app_config["llm_config"]["add_history"])
        self.top_k = int(app_config["llm_config"]["top_k"])
        self.top_p = float(app_config["llm_config"]["top_p"])
        self.max_new_tokens = int(app_config["llm_config"]["max_new_tokens"])
        self.do_sample = bool(app_config["llm_config"]["do_sample"])
        self.embedding_model = app_config["llm_config"]["embedding_model"]

        # Memory
        self.number_of_q_a_pairs = int(
            app_config["memory"]["number_of_q_a_pairs"])

        # clean up the upload doc vectordb if it exists
        self.create_directory(self.persist_directory)
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
