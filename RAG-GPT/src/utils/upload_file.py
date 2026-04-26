from typing import List, Tuple

from utils.prepare_vectordb import PrepareVectorDB
from utils.load_config import LoadConfig
from utils.summarizer import Summarizer

APPCFG = LoadConfig()


class UploadFile:
    @staticmethod
    def process_uploaded_files(
        files_dir: List,
        chatbot: List[dict],
        rag_with_dropdown: str,
    ) -> Tuple[str, List[dict]]:
        if rag_with_dropdown == "Upload doc: Process for RAG":
            prepare_vectordb_instance = PrepareVectorDB(
                data_directory=files_dir,
                persist_directory=APPCFG.custom_persist_directory,
                embedding_model_engine=APPCFG.embedding_model_engine,
                chunk_size=APPCFG.chunk_size,
                chunk_overlap=APPCFG.chunk_overlap,
            )
            prepare_vectordb_instance.prepare_and_save_vectordb()
            chatbot.append({
                "role": "assistant",
                "content": "Uploaded files are ready. Please ask your question",
            })
        elif rag_with_dropdown == "Upload doc: Give Full summary":
            final_summary = Summarizer.summarize_the_pdf(
                file_dir=files_dir[0],
                max_final_token=APPCFG.max_final_token,
                token_threshold=APPCFG.token_threshold,
                gpt_model=APPCFG.llm_engine,
                temperature=APPCFG.temperature,
                summarizer_llm_system_role=APPCFG.summarizer_llm_system_role,
                final_summarizer_llm_system_role=APPCFG.final_summarizer_llm_system_role,
                character_overlap=APPCFG.character_overlap,
            )
            chatbot.append({"role": "assistant", "content": final_summary})
        else:
            chatbot.append({
                "role": "assistant",
                "content": "If you would like to upload a PDF, please select your desired action in 'rag_with' dropdown.",
            })
        return "", chatbot
