import shutil
import gradio as gr
import time
import yaml
import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from typing import List, Tuple
import re
from dotenv import load_dotenv
import ast
import html

load_dotenv()

# Open AI configs
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

with open("configs/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)
# LLM configs
llm_engine = app_config["llm_config"]["engine"]
llm_system_role = app_config["llm_config"]["llm_system_role"]
# llm_temperature = app_config["llm_config"]["temperature"]
embedding_model = OpenAIEmbeddings()

# Retrieval configs
k = app_config["retrieval_config"]["k"]


class ChatBot:
    @staticmethod
    def respond(chatbot: List, message: str, data_type: str = "Preprocessed", temperature: float = 0.0) -> Tuple:
        if data_type == "Preprocessed" or data_type == [] or data_type == None:
            # directories
            persist_directory = app_config["directories"]["persist_directory"]
            vectordb = Chroma(persist_directory=persist_directory,
                              embedding_function=embedding_model)
        elif data_type == "Uploaded":
            custom_persist_directory = app_config["directories"]["custom_persist_directory"]
            vectordb = Chroma(persist_directory=custom_persist_directory,
                              embedding_function=embedding_model)

        question = "# User new question:\n" + message
        docs = vectordb.similarity_search(question, k=k)
        references = ChatBot.clean_references(docs)
        retrieved_docs_page_content = [
            str(x.page_content)+"\n\n" for x in docs]
        # retrieved_docs_links_and_page_numbers = [
        #     x.meta_data for x in docs]

        retrieved_docs_page_content = "# Retrieved content:\n" + \
            str(retrieved_docs_page_content)
        prompt = retrieved_docs_page_content + "\n\n" + question
        response = openai.ChatCompletion.create(
            engine=llm_engine,
            messages=[
                {"role": "system", "content": llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            stream=False
        )
        chatbot.append(
            (message, response["choices"][0]["message"]["content"]))
        time.sleep(2)

        return "", chatbot, references

    @staticmethod
    def feedback(data: gr.LikeData):
        if data.liked:
            print("You upvoted this response: " + data.value)
        else:
            print("You downvoted this response: " + data.value)

    @staticmethod
    def clean_references(documents: List) -> str:
        server_url = "http://localhost:8000"
        documents = [str(x)+"\n\n" for x in documents]
        markdown_documents = ""
        counter = 1
        for doc in documents:
            # print(doc)
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

            # Append cleaned content to the markdown string with two newlines between documents
            markdown_documents += f"Reference {counter}:\n" + content + "\n\n" + \
                f"Filename: {os.path.basename(metadata_dict['source'])}" + " | " +\
                f"Page number: {str(metadata_dict['page'])}" + " | " +\
                f"[View PDF]({pdf_url})" "\n\n"
            counter += 1

        return markdown_documents


class UISettings:
    @staticmethod
    def toggle_sidebar(state):
        state = not state
        return gr.update(visible=state), state


class GradioUploadFile:
    @staticmethod
    def check_directory(directory_path):
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            os.makedirs(directory_path)

    @staticmethod
    def process_uploaded_files(files_dir: List, chatbot):
        from prepare_vectordb import PrepareVectorDB
        embedding_model_engine = app_config["embedding_model_config"]["engine"]
        chunk_size = app_config["splitter_config"]["chunk_size"]
        chunk_overlap = app_config["splitter_config"]["chunk_overlap"]
        custom_persist_directory = app_config["directories"]["custom_persist_directory"]
        GradioUploadFile.check_directory(custom_persist_directory)
        prepare_vectordb_instance = PrepareVectorDB(data_directory=files_dir,
                                                    persist_directory=custom_persist_directory,
                                                    embedding_model_engine=embedding_model_engine,
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap)
        prepare_vectordb_instance.prepare_and_save_vectordb()
        chatbot.append(
            ("I just uploaded some documents.", "Uploaded files are ready. Please ask your question"))
        return "", chatbot
