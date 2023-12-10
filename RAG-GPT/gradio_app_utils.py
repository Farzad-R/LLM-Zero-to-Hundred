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
import ast
import html
from cfg import load_cfg

load_cfg()

with open("configs/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)
# LLM configs
llm_engine = app_config["llm_config"]["engine"]
llm_system_role = app_config["llm_config"]["llm_system_role"]
embedding_model = OpenAIEmbeddings()

# Retrieval configs
k = app_config["retrieval_config"]["k"]


class ChatBot:
    """
    Class representing a chatbot with document retrieval and response generation capabilities.

    This class provides static methods for responding to user queries, handling feedback, and
    cleaning references from retrieved documents.
    """
    @staticmethod
    def respond(chatbot: List, message: str, data_type: str = "Preprocessed", temperature: float = 0.0) -> Tuple:
        """
        Generate a response to a user query using document retrieval and language model completion.

        Parameters:
            - chatbot (List): List representing the chatbot's conversation history.
            - message (str): The user's query.
            - data_type (str): Type of data used for document retrieval ("Preprocessed" or "Uploaded").
            - temperature (float): Temperature parameter for language model completion.

        Returns:
            Tuple: A tuple containing an empty string, the updated chat history, and references from retrieved documents.
        """
        if data_type == "Preprocessed" or data_type == [] or data_type == None:
            # directories
            persist_directory = app_config["directories"]["persist_directory"]
            vectordb = Chroma(persist_directory=persist_directory,
                              embedding_function=embedding_model)
        elif data_type == "Uploaded":
            custom_persist_directory = app_config["directories"]["custom_persist_directory"]
            vectordb = Chroma(persist_directory=custom_persist_directory,
                              embedding_function=embedding_model)

        docs = vectordb.similarity_search(message, k=k)
        question = "# User new question:\n" + message
        references = ChatBot.clean_references(docs)
        retrieved_docs_page_content = [
            str(x.page_content)+"\n\n" for x in docs]
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
        """
        Process user feedback on the generated response.

        Parameters:
            data (gr.LikeData): Gradio LikeData object containing user feedback.
        """
        if data.liked:
            print("You upvoted this response: " + data.value)
        else:
            print("You downvoted this response: " + data.value)

    @staticmethod
    def clean_references(documents: List) -> str:
        """
        Clean and format references from retrieved documents.

        Parameters:
            documents (List): List of retrieved documents.

        Returns:
            str: A string containing cleaned and formatted references.

        Example:
        ```python
        references = ChatBot.clean_references(retrieved_documents)
        ```

        """
        server_url = "http://localhost:8000"
        documents = [str(x)+"\n\n" for x in documents]
        markdown_documents = ""
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

            # Append cleaned content to the markdown string with two newlines between documents
            markdown_documents += f"Reference {counter}:\n" + content + "\n\n" + \
                f"Filename: {os.path.basename(metadata_dict['source'])}" + " | " +\
                f"Page number: {str(metadata_dict['page'])}" + " | " +\
                f"[View PDF]({pdf_url})" "\n\n"
            counter += 1

        return markdown_documents


class UISettings:
    """
    Utility class for managing UI settings.

    This class provides static methods for toggling UI components, such as a sidebar.

    Example:
    ```python
    ui_state = True
    updated_ui, new_state = UISettings.toggle_sidebar(ui_state)
    ```
    """
    @staticmethod
    def toggle_sidebar(state):
        """
        Toggle the visibility state of a UI component.

        Parameters:
            state: The current state of the UI component.

        Returns:
            Tuple: A tuple containing the updated UI component state and the new state.

        Example:
        ```python
        ui_state = True
        updated_ui, new_state = UISettings.toggle_sidebar(ui_state)
        ```
        """
        state = not state
        return gr.update(visible=state), state


class GradioUploadFile:
    """
    Utility class for handling file uploads and processing.

    This class provides static methods for checking directories and processing uploaded files
    to prepare a VectorDB.

    Example:
    ```python
    files_dir = ['/path/to/uploaded/files']
    chatbot_instance = Chatbot()  # Assuming Chatbot is an existing class
    GradioUploadFile.process_uploaded_files(files_dir, chatbot_instance)
    ```
    """

    @staticmethod
    def check_directory(directory_path):
        """
        Check if a directory exists, and if it does, remove it and create a new one.

        Parameters:
            directory_path (str): The path of the directory to be checked and recreated.

        Example:
        ```python
        GradioUploadFile.check_directory("/path/to/directory")
        ```
        """
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            os.makedirs(directory_path)

    @staticmethod
    def process_uploaded_files(files_dir: List, chatbot: List) -> Tuple:
        """
        Process uploaded files to prepare a VectorDB.

        Parameters:
            - files_dir (List): List of paths to the uploaded files.
            - chatbot: An instance of the chatbot for communication.

        Returns:
            Tuple: A tuple containing an empty string and the updated chatbot instance.

        Example:
        ```python
        files_dir = ['/path/to/uploaded/files']
        chatbot_instance = Chatbot()  # Assuming Chatbot is an existing class
        GradioUploadFile.process_uploaded_files(files_dir, chatbot_instance)
        ```
        """
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
