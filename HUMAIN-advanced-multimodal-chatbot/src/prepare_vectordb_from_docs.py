
"""
To prepare the vectorDB from your documents:

Place your files in `data/docs` folder. Run this module in the terminal. A vectordb will be created from the documents in `data/vectordb/processed` directory.
"""

import os
from utils.raggpt.prepare_vectordb import PrepareVectorDB
from utils.raggpt.load_rag_config import LoadRAGConfig
from utils.ai_assistant.load_ai_assistant_config import LoadAIAssistantConfig

CONFIG = LoadRAGConfig()
APPCFG = LoadAIAssistantConfig()


def upload_data_manually() -> None:
    """
    Uploads data manually to the VectorDB.

    This function initializes a PrepareVectorDB instance with configuration parameters
    such as data_directory, persist_directory, embedding_model_engine, chunk_size,
    and chunk_overlap. It then checks if the VectorDB already exists in the specified
    persist_directory. If not, it calls the prepare_and_save_vectordb method to
    create and save the VectorDB. If the VectorDB already exists, a message is printed
    indicating its presence.

    Returns:
        None
    """
    prepare_vectordb_instance = PrepareVectorDB(
        data_directory=CONFIG.data_directory,
        persist_directory=CONFIG.persist_directory,
        embedding_model_engine=CONFIG.embedding_model_engine,
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )
    if not len(os.listdir(CONFIG.persist_directory)) != 0:
        prepare_vectordb_instance.prepare_and_save_vectordb()
    else:
        print(f"VectorDB already exists in {CONFIG.persist_directory}")
    return None


if __name__ == "__main__":
    upload_data_manually()
