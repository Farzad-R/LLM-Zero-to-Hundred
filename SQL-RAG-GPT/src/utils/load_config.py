
import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
import shutil
from langchain.chat_models import AzureChatOpenAI

print("Environment variables are loaded:", load_dotenv())


class LoadConfig:
    def __init__(self) -> None:
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # directories
        self.stored_csv_xlsx_directory = here(
            app_config["directories"]["stored_csv_xlsx_directory"])
        self.sqldb_directory = str(here(
            app_config["directories"]["sqldb_directory"]))
        self.uploaded_files_sqldb_directory = str(here(
            app_config["directories"]["uploaded_files_sqldb_directory"]))
        self.stored_csv_xlsx_sqldb_directory = str(here(
            app_config["directories"]["stored_csv_xlsx_sqldb_directory"]))

        # LLM configs
        self.model_name = app_config["llm_config"]["engine"]
        self.llm_system_role = app_config["llm_config"]["llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]

        # Load Azure OpenAI's GPT model
        self.load_gpt()

        # Clean up the upload csv SQL DB if it exists
        # self.remove_directory(self.csv_to_sqldb_directory)

    def load_gpt(self):
        azure_openai_api_key = os.environ["OPENAI_API_KEY"]
        azure_openai_endpoint = os.environ["OPENAI_API_BASE"]
        self.llm = AzureChatOpenAI(
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            azure_deployment=self.model_name,
            model_name=self.model_name,
            temperature=self.temperature)

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
