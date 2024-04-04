from utils.raggpt.prepare_vectordb import PrepareVectorDB
from typing import List, Tuple
from utils.raggpt.load_rag_config import LoadRAGConfig
from utils.raggpt.summarizer import Summarizer
import gradio as gr
RAGCFG = LoadRAGConfig()


class UploadFile:
    """
    Utility class for handling file uploads and processing.

    This class provides static methods for checking directories and processing uploaded files
    to prepare a VectorDB.
    """

    @staticmethod
    def process_uploaded_files(files_dir: List, chatbot: List, app_functionality: str) -> Tuple:
        """
        Process uploaded files to prepare a VectorDB.

        Parameters:
            files_dir (List): List of paths to the uploaded files.
            chatbot (List): An instance of the chatbot for communication. A list of tuples containing the chat history.
            app_functionality (str): The functionality chosen by the user.

        Returns:
            Tuple: A tuple containing an empty string and the updated chatbot instance.
        """
        if app_functionality == "RAG-GPT: RAG with upload documents":
            prepare_vectordb_instance = PrepareVectorDB(data_directory=files_dir,
                                                        persist_directory=RAGCFG.custom_persist_directory,
                                                        embedding_model_engine=RAGCFG.embedding_model_engine,
                                                        chunk_size=RAGCFG.chunk_size,
                                                        chunk_overlap=RAGCFG.chunk_overlap)
            prepare_vectordb_instance.prepare_and_save_vectordb()
            chatbot.append(
                (None, "Uploaded files are ready. Please ask your question"))
        elif app_functionality == "RAG-GPT: Summarize a document":
            final_summary = Summarizer.summarize_the_pdf(file_dir=files_dir[0],
                                                         max_final_token=RAGCFG.max_final_token,
                                                         token_threshold=RAGCFG.token_threshold,
                                                         gpt_model=RAGCFG.llm_engine,
                                                         temperature=RAGCFG.temperature,
                                                         summarizer_llm_system_role=RAGCFG.summarizer_llm_system_role,
                                                         final_summarizer_llm_system_role=RAGCFG.final_summarizer_llm_system_role,
                                                         character_overlap=RAGCFG.character_overlap)
            chatbot.append(
                (None, final_summary))
        else:
            chatbot.append(
                (None, "If you would like to upload a PDF, please select your desired action in `app_functionality` dropdown."))
        return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"])
