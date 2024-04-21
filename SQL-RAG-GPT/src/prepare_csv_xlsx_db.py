from utils.prepare_db_from_csv_xlsx import PrepareSQLFromCSVs
from utils.load_config import LoadConfig

APPCFG = LoadConfig()

if __name__ == "__main__":
    prep_sql_instance = PrepareSQLFromCSVs(APPCFG.stored_csv_xlsx_directory)
    prep_sql_instance.run_pipeline()
