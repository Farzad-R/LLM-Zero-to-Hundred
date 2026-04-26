import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


def _make_doc(content: str = "doc content") -> Document:
    return Document(page_content=content, metadata={"source": "test.pdf", "page": 0})


def test_prepare_and_save_vectordb_calls_from_documents(tmp_path) -> None:
    """prepare_and_save_vectordb() calls Chroma.from_documents with chunked docs."""
    from utils.prepare_vectordb import PrepareVectorDB

    fake_docs = [_make_doc("page 1"), _make_doc("page 2")]
    mock_vectordb = MagicMock()
    mock_vectordb._collection.count.return_value = 2

    with patch("utils.prepare_vectordb.PyPDFLoader") as mock_loader, \
         patch("utils.prepare_vectordb.Chroma") as mock_chroma, \
         patch("utils.prepare_vectordb.OpenAIEmbeddings"), \
         patch("os.listdir", return_value=["doc.pdf"]):
        mock_loader.return_value.load.return_value = fake_docs
        mock_chroma.from_documents.return_value = mock_vectordb

        instance = PrepareVectorDB(
            data_directory=str(tmp_path),
            persist_directory=str(tmp_path / "vectordb"),
            embedding_model_engine="text-embedding-ada-002",
            chunk_size=1500,
            chunk_overlap=500,
        )
        result = instance.prepare_and_save_vectordb()

    mock_chroma.from_documents.assert_called_once()
    call_kwargs = mock_chroma.from_documents.call_args[1]
    assert "documents" in call_kwargs
    assert "embedding" in call_kwargs
    assert call_kwargs["persist_directory"] == str(tmp_path / "vectordb")
    assert result is mock_vectordb


def test_chunk_sizes_applied_from_config() -> None:
    """Text splitter is initialised with the chunk_size and chunk_overlap from the caller."""
    from utils.prepare_vectordb import PrepareVectorDB

    with patch("utils.prepare_vectordb.OpenAIEmbeddings"):
        instance = PrepareVectorDB(
            data_directory="/some/dir",
            persist_directory="/some/vectordb",
            embedding_model_engine="text-embedding-ada-002",
            chunk_size=800,
            chunk_overlap=200,
        )

    assert instance.text_splitter._chunk_size == 800
    assert instance.text_splitter._chunk_overlap == 200
