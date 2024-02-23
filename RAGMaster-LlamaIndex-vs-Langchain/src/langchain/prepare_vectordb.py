import os
from langchain_utils.langchain_index_utils import PrepareVectorDB
from langchain_utils.load_config import LoadConfig
import openai
from dotenv import load_dotenv
load_dotenv()

CFG = LoadConfig()
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")
print("===================")


def prep_langchain_vdb():
    prepare_vectordb_instance = PrepareVectorDB(data_directory=CFG.documents_dir,
                                                token_vector_db_save_dir=CFG.token_vector_db_save_dir,
                                                recursive_vector_db_save_dir=CFG.recursive_vector_db_save_dir,
                                                embedding_model_engine=CFG.embed_model_name,
                                                chunk_size=CFG.langchain_recursive_chunk_size,
                                                chunk_overlap=CFG.langchain_recursive_chunk_overlap,
                                                token_chunk_size=CFG.langchain_token_chunk_size,
                                                token_chunk_overlap=CFG.langchain_token_chunk_overlap,
                                                splitter_type=CFG.splitter_type)

    prepare_vectordb_instance.prepare_and_save_vectordb()
    return None


if __name__ == "__main__":
    prep_langchain_vdb()
