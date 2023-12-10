from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
import traceback

load_dotenv()


class PrepareURLVectorDB:
    """
    A class for preparing and saving a VectorDB using OpenAI embeddings.

    This class facilitates the process of loading documents, chunking them, and creating a VectorDB
    with OpenAI embeddings. It provides methods to prepare and save the VectorDB.

    Parameters:
        - data_directory (str or List[str]): The directory or list of directories containing the documents.
        - persist_directory (str): The directory to save the VectorDB.
        - embedding_model_engine (str): The engine for OpenAI embeddings.
        - chunk_size (int): The size of the chunks for document processing.
        - chunk_overlap (int): The overlap between chunks.

    Example:
    ```python
    vector_db_creator = PrepareVectorDB(
        data_directory='path/to/documents',
        persist_directory='path/to/vectordb',
        embedding_model_engine='openai-gpt-3.5-turbo',
        chunk_size=500,
        chunk_overlap=100
    )
    vectordb = vector_db_creator.prepare_and_save_vectordb()
    ```

    """

    def __init__(
            self,
            url: str,
            persist_directory: str,
            embedding_model_engine: str,
            chunk_size: int,
            chunk_overlap: int
    ) -> None:
        """
        Initialize the PrepareVectorDB instance.
        NOTE: __Cannot handle multiple urls for now__ 

        Parameters:
        - url (str):  The requested url.
        - persist_directory (str): The directory to save the VectorDB.
        - embedding_model_engine (str): The engine for OpenAI embeddings.
        - chunk_size (int): The size of the chunks for document processing.
        - chunk_overlap (int): The overlap between chunks.

        """

        self.embedding_model_engine = embedding_model_engine
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        """Other options: CharacterTextSplitter, NotionDirectoryLoader, TokenTextSplitter, etc."""
        self.url = url
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings()

    def _load_web_page(self):
        # I assumed that the user will only ask fo one url and it is a webpage and not a youtube link.
        try:
            loader = WebBaseLoader(self.url)
            documents = loader.load()
            return documents
        except:
            raise "The requested link is not supported by langchain yet."

    def __chunk_documents(self, docs: List) -> List:
        """
        Chunk the loaded documents using the specified text splitter.

        Parameters:
            - docs (List): The list of loaded documents.

        Returns:
            List: A list of chunked documents.

        """
        print("Chunking the webpage...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks:", len(chunked_documents), "\n\n")
        return chunked_documents

    def prepare_and_save_vectordb(self):
        """
        Load, chunk, and create a VectorDB with OpenAI embeddings, and save it.

        Returns:
            Chroma: The created VectorDB.
        """
        try:
            docs = self._load_web_page()
            chunked_documents = self.__chunk_documents(docs)
            print("Preparing vectordb...")
            vectordb = Chroma.from_documents(
                documents=chunked_documents,
                embedding=self.embedding,
                persist_directory=self.persist_directory
            )
            print("VectorDB is created and saved.")
            print("Number of vectors in vectordb:",
                  vectordb._collection.count(), "\n\n")
            return True
        except BaseException as e:
            print(f"Caught exception in PrepareURLVectorDB class: {e}")
            traceback.print_exc()
            return False
