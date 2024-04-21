import os
from typing import List, Tuple
from utils.load_config import LoadConfig
from sqlalchemy import create_engine, inspect
import pandas as pd
APPCFG = LoadConfig()


class ProcessFiles:
    def __init__(self, files_dir: List, chatbot: List) -> None:
        APPCFG = LoadConfig()
        self.files_dir = files_dir
        self.chatbot = chatbot
        db_path = APPCFG.uploaded_files_sqldb_directory
        db_path = f"sqlite:///{db_path}"
        self.engine = create_engine(db_path)
        print("Number of uploaded files:", len(self.files_dir))

    def process_uploaded_files(self) -> Tuple:
        for file_dir in self.files_dir:
            file_names_with_extensions = os.path.basename(file_dir)
            file_name, file_extension = os.path.splitext(
                file_names_with_extensions)
            if file_extension == ".csv":
                df = pd.read_csv(file_dir)
            elif file_extension == ".xlsx":
                df = pd.read_excel(file_dir)
            else:
                raise ValueError("The selected file type is not supported")
            df.to_sql(file_name, self.engine, index=False)
        print("==============================")
        print("All csv/xlsx files are saved into the sql database.")
        self.chatbot.append(
            (" ", "Uploaded files are ready. Please ask your question"))
        return "", self.chatbot

    def validate_db(self):
        insp = inspect(self.engine)
        table_names = insp.get_table_names()
        print("==============================")
        print("Available table nasmes in created SQL DB:", table_names)
        print("==============================")

    def run(self):
        input_txt, chatbot = self.process_uploaded_files()
        self.validate_db()
        return input_txt, chatbot


class UploadFile:
    @staticmethod
    def run_pipeline(files_dir: List, chatbot: List, chatbot_functionality: str):
        if chatbot_functionality == "Process files":
            pipeline_instance = ProcessFiles(
                files_dir=files_dir, chatbot=chatbot)
            input_txt, chatbot = pipeline_instance.run()
            return input_txt, chatbot
        else:
            pass
