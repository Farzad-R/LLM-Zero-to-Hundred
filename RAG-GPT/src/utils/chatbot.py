import os
import re
import ast
import html
import time
from typing import List, Tuple

from langchain_chroma import Chroma
from utils.load_config import LoadConfig

APPCFG = LoadConfig()
URL = "https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/RAG-GPT"
hyperlink = f"[RAG-GPT user guideline]({URL})"


class ChatBot:
    @staticmethod
    def respond(
        chatbot: List[dict],
        message: str,
        data_type: str = "Preprocessed doc",
        temperature: float = 0.0,
    ) -> Tuple[str, List[dict], str | None]:
        if data_type == "Preprocessed doc":
            db_file = os.path.join(APPCFG.persist_directory, "chroma.sqlite3")
            if os.path.exists(db_file):
                vectordb = Chroma(
                    persist_directory=APPCFG.persist_directory,
                    embedding_function=APPCFG.embedding_model,
                )
            else:
                chatbot.append({
                    "role": "assistant",
                    "content": f"VectorDB does not exist. Please first execute the 'upload_data_manually.py' module. For further information please visit {hyperlink}.",
                })
                return "", chatbot, None

        elif data_type == "Upload doc: Process for RAG":
            db_file = os.path.join(APPCFG.custom_persist_directory, "chroma.sqlite3")
            if os.path.exists(db_file):
                vectordb = Chroma(
                    persist_directory=APPCFG.custom_persist_directory,
                    embedding_function=APPCFG.embedding_model,
                )
            else:
                chatbot.append({
                    "role": "assistant",
                    "content": "No file was uploaded. Please first upload your files using the 'upload' button.",
                })
                return "", chatbot, None

        docs = vectordb.similarity_search(message, k=APPCFG.k)
        print(docs)
        question = "# User new question:\n" + message
        retrieved_content = ChatBot.clean_references(docs)

        # Memory: include the last N Q&A pairs from dict-format history
        recent = chatbot[-(APPCFG.number_of_q_a_pairs * 2):]
        chat_history = "Chat history:\n" + "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        ) + "\n\n"

        prompt = f"{chat_history}{retrieved_content}{question}"
        print("========================")
        print(prompt)

        response = APPCFG.openai_client.chat.completions.create(
            model=APPCFG.llm_engine,
            messages=[
                {"role": "system", "content": APPCFG.llm_system_role},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )

        chatbot.append({"role": "user", "content": message})
        chatbot.append({"role": "assistant", "content": response.choices[0].message.content})
        time.sleep(2)

        return "", chatbot, retrieved_content

    @staticmethod
    def clean_references(documents: List) -> str:
        server_url = "http://localhost:8000"
        documents = [str(x) + "\n\n" for x in documents]
        markdown_documents = ""
        counter = 1
        for doc in documents:
            content, metadata = re.match(
                r"page_content=(.*?)( metadata=\{.*\})", doc
            ).groups()
            metadata = metadata.split("=", 1)[1]
            metadata_dict = ast.literal_eval(metadata)

            content = bytes(content, "utf-8").decode("unicode_escape")
            content = re.sub(r"\\n", "\n", content)
            content = re.sub(r"\s*<EOS>\s*<pad>\s*", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
            content = html.unescape(content)
            content = content.encode("latin1").decode("utf-8", "ignore")
            content = re.sub(r"â", "-", content)
            content = re.sub(r"â", "∈", content)
            content = re.sub(r"Ã", "×", content)
            content = re.sub(r"ï¬", "fi", content)
            content = re.sub(r"Â·", "·", content)
            content = re.sub(r"ï¬", "fl", content)

            pdf_url = f"{server_url}/{os.path.basename(metadata_dict['source'])}"
            markdown_documents += (
                f"# Retrieved content {counter}:\n{content}\n\n"
                f"Source: {os.path.basename(metadata_dict['source'])} | "
                f"Page number: {metadata_dict['page']} | "
                f"[View PDF]({pdf_url})\n\n"
            )
            counter += 1

        return markdown_documents
