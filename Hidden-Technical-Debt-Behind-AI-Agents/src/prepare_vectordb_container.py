# https://docs.trychroma.com/production/containers/docker
import os
from typing import List
from chromadb import HttpClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.load_config import LoadConfig
from dotenv import load_dotenv

load_dotenv()


class PrepareVectorDB:
    """
    A class for preparing and uploading documents to a Chroma vector DB server using OpenAI embeddings.
    """

    def __init__(
        self,
        data_directory: str,
        collection_name: str,
        embedding_model_engine: str,
        chunk_size: int,
        chunk_overlap: int,
        chroma_host: str = "localhost",
        chroma_port: int = 8000
    ) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.data_directory = data_directory
        self.collection_name = collection_name
        self.embedding_fn = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=embedding_model_engine
        )
        self.client = HttpClient(host=chroma_host, port=chroma_port)
        self.client.heartbeat()

    def __load_all_documents(self) -> List:
        doc_counter = 0
        print("Loading documents...")
        docs = []
        document_list = os.listdir(self.data_directory)
        for doc_name in document_list:
            docs.extend(PyPDFLoader(os.path.join(
                self.data_directory, doc_name)).load())
            doc_counter += 1
        print("Number of loaded documents:", doc_counter)
        print("Number of pages:", len(docs), "\n\n")
        return docs

    def __chunk_documents(self, docs: List) -> List:
        print("Chunking documents...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks:", len(chunked_documents), "\n\n")
        return chunked_documents

    def prepare_and_save_vectordb(self):
        docs = self.__load_all_documents()
        chunked_docs = self.__chunk_documents(docs)
        texts = [d.page_content for d in chunked_docs]
        metadatas = [d.metadata for d in chunked_docs]

        print("Uploading to Chroma...")
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=[f"doc_{i}" for i in range(len(texts))]
        )

        print("✅ Upload complete. Number of vectors:",
              collection.count(), "\n\n")
        return collection


if __name__ == "__main__":
    CFG = LoadConfig()
    prepare_vectordb_instance = PrepareVectorDB(
        data_directory=CFG.data_directory,
        collection_name=CFG.collection_name,
        embedding_model_engine=CFG.embedding_model,
        chunk_size=CFG.chunk_size,
        chunk_overlap=CFG.chunk_overlap,
        chroma_host="chroma",  # Docker Compose service name
        chroma_port=8000
    )
    prepare_vectordb_instance.prepare_and_save_vectordb()
