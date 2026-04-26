import pytest
import os
from unittest.mock import patch, MagicMock


def test_respond_missing_vectordb() -> None:
    """When chroma.sqlite3 doesn't exist, an error is appended to chat history."""
    from utils.chatbot import ChatBot

    chatbot: list = []
    with patch("utils.chatbot.os.path.exists", return_value=False):
        result = ChatBot.respond(chatbot, "hello", data_type="Preprocessed doc")

    assert result[0] == ""
    assert len(result[1]) == 1
    assert "VectorDB does not exist" in result[1][0]["content"]
    assert result[2] is None


def test_respond_preprocessed_doc(mock_openai_client: MagicMock, mock_vectordb: MagicMock) -> None:
    """respond() returns (empty_str, updated_chatbot, retrieved_content) on success."""
    from utils import chatbot as chatbot_module
    from utils.chatbot import ChatBot

    with patch("utils.chatbot.os.path.exists", return_value=True), \
         patch("utils.chatbot.Chroma", return_value=mock_vectordb), \
         patch("utils.chatbot.ChatBot.clean_references", return_value="mock references"), \
         patch.object(chatbot_module.APPCFG, "openai_client", mock_openai_client):

        result = ChatBot.respond([], "What is RAG?", data_type="Preprocessed doc", temperature=0.0)

    empty_str, history, references = result
    assert empty_str == ""
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "What is RAG?"}
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Mock LLM response"
    assert references is not None


def test_respond_uses_model_from_config(mock_openai_client: MagicMock, mock_vectordb: MagicMock) -> None:
    """The model kwarg passed to the OpenAI client matches the config value."""
    from utils import chatbot as chatbot_module
    from utils.chatbot import ChatBot

    with patch("utils.chatbot.os.path.exists", return_value=True), \
         patch("utils.chatbot.Chroma", return_value=mock_vectordb), \
         patch("utils.chatbot.ChatBot.clean_references", return_value="mock references"), \
         patch.object(chatbot_module.APPCFG, "openai_client", mock_openai_client):

        ChatBot.respond([], "test", data_type="Preprocessed doc")

    call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == chatbot_module.APPCFG.llm_engine


def test_respond_upload_doc_missing_vectordb() -> None:
    """When custom chroma.sqlite3 doesn't exist, an error is appended."""
    from utils.chatbot import ChatBot

    chatbot: list = []
    with patch("utils.chatbot.os.path.exists", return_value=False):
        result = ChatBot.respond(chatbot, "hello", data_type="Upload doc: Process for RAG")

    assert "No file was uploaded" in result[1][0]["content"]
    assert result[2] is None
