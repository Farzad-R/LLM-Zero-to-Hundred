
import os


def create_directory(directory_path):
    """
    Create a directory if it does not exist.

    Parameters:
        directory_path (str): The path of the directory to be created.

    Example:
    ```python
    create_directory("/path/to/new/directory")
    ```

    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def upload_data_manually() -> None:
    """
    Upload data manually using OpenAI API and prepare a VectorDB.

    This function initializes OpenAI API credentials, loads configuration from a YAML file,
    creates necessary directories, and prepares a VectorDB instance using the specified parameters.

    Example:
    ```python
    upload_data_manually()
    ```

    Returns:
    None

    """
    import openai
    from dotenv import load_dotenv
    from prepare_vectordb import PrepareVectorDB
    import os
    import yaml
    load_dotenv()
    openai.api_version = os.getenv("OPENAI_API_VERSION")
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    with open("configs/app_config.yml") as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)
    embedding_model_engine = app_config["embedding_model_config"]["engine"]
    chunk_size = app_config["splitter_config"]["chunk_size"]
    chunk_overlap = app_config["splitter_config"]["chunk_overlap"]
    data_directory = app_config["directories"]["data_directory"]
    persist_directory = app_config["directories"]["persist_directory"]
    create_directory(persist_directory)
    prepare_vectordb_instance = PrepareVectorDB(
        data_directory=data_directory,
        persist_directory=persist_directory,
        embedding_model_engine=embedding_model_engine,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not len(os.listdir(persist_directory)) != 0:
        prepare_vectordb_instance.prepare_and_save_vectordb()
    else:
        print(f"VectorDB already exists in {persist_directory}")
    return None


if __name__ == "__main__":
    upload_data_manually()
