"""
Document Vectorstore Module with ChromaDB
Manages the document collection for RAG retrieval with persistent storage.
"""

from typing import List, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


class DocumentVectorStore:
    """Manages document vectorstore for RAG retrieval using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "taskflow_docs",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the document vectorstore with ChromaDB.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection in ChromaDB
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vectorstore: Optional[Chroma] = None

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

    def load_existing(self) -> bool:
        """
        Load existing ChromaDB collection if it exists.

        Returns:
            True if collection exists and was loaded, False otherwise
        """
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

            # Check if collection has documents
            collection = self.vectorstore._collection
            count = collection.count()

            if count > 0:
                print(f"✅ Loaded existing collection with {count} documents")
                return True
            else:
                print("ℹ️  Collection exists but is empty")
                return False

        except Exception as e:
            print(f"ℹ️  No existing collection found: {e}")
            return False

    def create_from_documents(self, documents: List[Document], clear_existing: bool = False):
        """
        Create vectorstore from documents.

        Args:
            documents: List of Document objects
            clear_existing: If True, delete existing collection before creating new one
        """
        print(f"📥 Processing {len(documents)} documents...")

        # Split documents
        print("✂️  Splitting documents into chunks...")
        doc_splits = self.text_splitter.split_documents(documents)
        print(f"  ✓ Created {len(doc_splits)} chunks")

        # Delete existing collection if requested
        if clear_existing and self.vectorstore is not None:
            print("🗑️  Clearing existing collection...")
            try:
                self.vectorstore.delete_collection()
            except:
                pass

        # Create vectorstore
        print("🔢 Creating embeddings and ChromaDB collection...")
        self.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )

        print(f"  ✓ ChromaDB collection created at: {self.persist_directory}")
        print(f"  ✓ Collection name: {self.collection_name}")

    def add_documents(self, documents: List[Document]):
        """
        Add documents to existing vectorstore.

        Args:
            documents: List of Document objects to add
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vectorstore not initialized. Create or load first.")

        print(f"📥 Adding {len(documents)} documents...")

        # Split documents
        doc_splits = self.text_splitter.split_documents(documents)
        print(f"  ✓ Created {len(doc_splits)} chunks")

        # Add to vectorstore
        self.vectorstore.add_documents(doc_splits)
        print("  ✓ Documents added to ChromaDB")

    def get_retriever(self, k: int = 4, search_type: str = "similarity"):
        """
        Get a retriever for the vectorstore.

        Args:
            k: Number of documents to retrieve
            search_type: Type of search ("similarity" or "mmr")

        Returns:
            Retriever object
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vectorstore not initialized. Load or create documents first.")

        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vectorstore not initialized. Load or create documents first.")

        return self.vectorstore.similarity_search(query, k=k)

    def get_stats(self) -> dict:
        """
        Get statistics about the vectorstore.

        Returns:
            Dictionary with stats
        """
        if self.vectorstore is None:
            return {"status": "not_initialized"}

        try:
            collection = self.vectorstore._collection
            count = collection.count()

            return {
                "status": "initialized",
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def delete_collection(self):
        """Delete the entire collection."""
        if self.vectorstore is not None:
            print(f"🗑️  Deleting collection: {self.collection_name}")
            self.vectorstore.delete_collection()
            self.vectorstore = None
            print("  ✓ Collection deleted")
