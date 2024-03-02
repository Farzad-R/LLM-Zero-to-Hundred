import gradio as gr
import time
import os
from langchain.vectorstores import Chroma
from typing import List, Tuple
import re
import ast
import html
from utils.load_config import LoadConfig
from langchain.embeddings import HuggingFaceEmbeddings
import requests
import torch
FLASK_APP_ENDPOINT = "http://127.0.0.1:8888/generate_text"

APPCFG = LoadConfig()
URL = "https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/RAG-GPT"
hyperlink = f"[RAG-GPT user guideline]({URL})"


class ChatBot:
    """
    Class representing a chatbot with document retrieval and response generation capabilities.

    This class provides static methods for responding to user queries, handling feedback, and
    cleaning references from retrieved documents.
    """
    @staticmethod
    def respond(chatbot: List,
                message: str,
                data_type: str = "Preprocessed doc",
                temperature: float = 0.1,
                top_k: int = 10,
                top_p: float = 0.1) -> Tuple:
        """
        Generate a response to a user query using document retrieval and language model completion.

        Parameters:
            chatbot (List): List representing the chatbot's conversation history.
            message (str): The user's query.
            data_type (str): Type of data used for document retrieval ("Preprocessed doc" or "Upload doc: Process for RAG").
            temperature (float): Temperature parameter for language model completion.

        Returns:
            Tuple: A tuple containing an empty string, the updated chat history, and references from retrieved documents.
        """

        # Retrieve embedding function from code env resources
        # emb_model = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            # cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
        )
        if data_type == "Preprocessed doc":
            # directories
            if os.path.exists(APPCFG.persist_directory):
                vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                                  embedding_function=embedding_function)
            else:
                chatbot.append(
                    (message, f"VectorDB does not exist. Please first execute the 'upload_data_manually.py' module. For further information please visit {hyperlink}."))
                return "", chatbot, None

        elif data_type == "Upload doc: Process for RAG":
            if os.path.exists(APPCFG.custom_persist_directory):
                vectordb = Chroma(persist_directory=APPCFG.custom_persist_directory,
                                  embedding_function=embedding_function)
            else:
                chatbot.append(
                    (message, f"No file was uploaded. Please first upload your files using the 'upload' button."))
                return "", chatbot, None

        docs = vectordb.similarity_search(message, k=APPCFG.k)
        question = "# Prompt that you have to answer:\n" + message
        retrieved_content, markdown_documents = ChatBot.clean_references(docs)
        # Memory: previous two Q&A pairs
        chat_history = f"Chat history:\n {str(chatbot[-APPCFG.number_of_q_a_pairs:])}\n\n"
        if APPCFG.add_history:
            prompt_wrapper = f"{APPCFG.llm_system_role_with_history}\n\n{chat_history}\n\n{retrieved_content}{question}"
        else:
            prompt_wrapper = f"{APPCFG.llm_system_role_without_history}\n\n{question}\n\n{retrieved_content}"

        print("========================")
        print(prompt_wrapper)
        print("========================")
        messages = [
            {"role": "user", "content": prompt_wrapper},
        ]
        data = {
            "prompt": messages,
            "max_new_tokens": APPCFG.max_new_tokens,
            "do_sample": APPCFG.do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }
        response = requests.post(FLASK_APP_ENDPOINT, json=data)
        # print(response.text)
        response_json = response.json()

        chatbot.append(
            (message, response_json["response"]))
        # Clean up GPU memory
        del vectordb
        del docs
        torch.cuda.empty_cache()
        return "", chatbot, markdown_documents

    @staticmethod
    def clean_references(documents: List) -> str:
        """
        Clean and format references from retrieved documents.

        Parameters:
            documents (List): List of retrieved documents.

        Returns:
            str: A string containing cleaned and formatted references.
        """
        server_url = "http://localhost:8000"
        documents = [str(x)+"\n\n" for x in documents]
        markdown_documents = ""
        retrieved_content = ""
        counter = 1
        for doc in documents:
            # Extract content and metadata
            content, metadata = re.match(
                r"page_content=(.*?)( metadata=\{.*\})", doc).groups()
            metadata = metadata.split('=', 1)[1]
            metadata_dict = ast.literal_eval(metadata)

            # Decode newlines and other escape sequences
            content = bytes(content, "utf-8").decode("unicode_escape")

            # Replace escaped newlines with actual newlines
            content = re.sub(r'\\n', '\n', content)
            # Remove special tokens
            content = re.sub(r'\s*<EOS>\s*<pad>\s*', ' ', content)
            # Remove any remaining multiple spaces
            content = re.sub(r'\s+', ' ', content).strip()

            # Decode HTML entities
            content = html.unescape(content)

            # Replace incorrect unicode characters with correct ones
            content = content.encode('latin1').decode('utf-8', 'ignore')

            # Remove or replace special characters and mathematical symbols
            # This step may need to be customized based on the specific symbols in your documents
            content = re.sub(r'â', '-', content)
            content = re.sub(r'â', '∈', content)
            content = re.sub(r'Ã', '×', content)
            content = re.sub(r'ï¬', 'fi', content)
            content = re.sub(r'â', '∈', content)
            content = re.sub(r'Â·', '·', content)
            content = re.sub(r'ï¬', 'fl', content)

            pdf_url = f"{server_url}/{os.path.basename(metadata_dict['source'])}"
            retrieved_content += f"# Content {counter}:\n" + \
                content + "\n\n"

            # Append cleaned content to the markdown string with two newlines between documents
            markdown_documents += f"# Retrieved content {counter}:\n" + content + "\n\n" + \
                f"Source: {os.path.basename(metadata_dict['source'])}" + " | " +\
                f"Page number: {str(metadata_dict['page'])}" + " | " +\
                f"[View PDF]({pdf_url})" "\n\n"
            counter += 1

        return retrieved_content, markdown_documents
