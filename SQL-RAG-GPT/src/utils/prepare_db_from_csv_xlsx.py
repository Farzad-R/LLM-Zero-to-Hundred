import os
import pandas as pd
from utils.load_config import LoadConfig
import pandas as pd
from sqlalchemy import create_engine, inspect


class PrepareSQLFromCSVs:
    def __init__(self, files_dir) -> None:
        APPCFG = LoadConfig()
        self.files_directory = files_dir
        self.file_dir_list = os.listdir(files_dir)
        db_path = APPCFG.stored_csv_xlsx_sqldb_directory
        db_path = f"sqlite:///{db_path}"
        self.engine = create_engine(db_path)
        print("Number of csv files:", len(self.file_dir_list))

    def _prepare_db(self):
        for file in self.file_dir_list:
            full_file_path = os.path.join(self.files_directory, file)
            file_name, file_extension = os.path.splitext(file)
            if file_extension == ".csv":
                df = pd.read_csv(full_file_path)
            elif file_extension == ".xlsx":
                df = pd.read_excel(full_file_path)
            else:
                raise ValueError("The selected file type is not supported")
            df.to_sql(file_name, self.engine, index=False)
        print("==============================")
        print("All csv files are saved into the sql database.")

    def _validate_db(self):
        insp = inspect(self.engine)
        table_names = insp.get_table_names()
        print("==============================")
        print("Available table nasmes in created SQL DB:", table_names)
        print("==============================")

    def run_pipeline(self):
        self._prepare_db()
        self._validate_db()
