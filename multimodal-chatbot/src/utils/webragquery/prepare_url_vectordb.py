from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
import traceback


class PrepareURLVectorDB:
    """
    A class for preparing and saving a VectorDB using OpenAI embeddings.

    This class facilitates the process of loading documents, chunking them, and creating a VectorDB
    with OpenAI embeddings. It provides methods to prepare and save the VectorDB.

    Args:
        data_directory (str or List[str]): The directory or list of directories containing the documents.
        persist_directory (str): The directory to save the VectorDB.
        embedding_model_engine (str): The engine for OpenAI embeddings.
        chunk_size (int): The size of the chunks for document processing.
        chunk_overlap (int): The overlap between chunks.
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
        Initialize the PrepareVectorDB instance. It does not handle multiple urls for now.

        Args:
            url (str):  The requested url.
            persist_directory (str): The directory to save the VectorDB.
            embedding_model_engine (str): The engine for OpenAI embeddings.
            chunk_size (int): The size of the chunks for document processing.
            chunk_overlap (int): The overlap between chunks.
        """

        self.embedding_model_engine = embedding_model_engine
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "]
        )
        """Other options: CharacterTextSplitter, NotionDirectoryLoader, TokenTextSplitter, etc."""
        self.url = self._ensure_https(url)
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings()

    def _ensure_https(self, url: str) -> str:
        if not url.startswith("https://"):
            url = "https://" + url
        return url

    def _load_web_page(self):
        """
        Load web page content from the specified URL.

        Returns:
            List[Document]: A list of documents representing the content of the web page.

        Raises:
            Exception: If the requested link is not supported by LangChain.

        Note:
            This method assumes that the user will provide a single URL pointing to a webpage (not a YouTube link).
            It utilizes a WebBaseLoader instance to load and extract content from the specified URL.
        """
        try:
            loader = WebBaseLoader(self.url)
            documents = loader.load()
            return documents
        except Exception as e:
            raise Exception(
                "The requested link is not supported by langchain yet. Error: {}".format(e))

    def __chunk_documents(self, docs: List) -> List:
        """
        Chunk the loaded documents using the specified text splitter.

        Args:
            docs (List): The list of loaded documents.

        Returns:
            List: A list of chunked documents.
        """
        print("Chunking the webpage...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks:", len(chunked_documents), "\n\n")
        return chunked_documents

    def prepare_and_save_vectordb(self) -> bool:
        """
        Load, chunk, and create a VectorDB with OpenAI embeddings, and save it.

        Returns:
            bool: True, if vectorDB creation was successful. False, if the task was failed.
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
