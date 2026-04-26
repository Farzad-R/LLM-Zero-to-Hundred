"""
Phase B tests: verify Gradio 5 dict-format chat history throughout the pipeline.
All chatbot.append() calls must produce {"role": ..., "content": ...} dicts,
not (user, bot) tuples.
"""
import pytest
from unittest.mock import patch, MagicMock


def _is_user_message(msg: dict, expected_content: str) -> bool:
    return msg.get("role") == "user" and expected_content in msg.get("content", "")


def _is_assistant_message(msg: dict) -> bool:
    return msg.get("role") == "assistant" and "content" in msg


def test_chatbot_respond_appends_dict_messages(
    mock_openai_client: MagicMock, mock_vectordb: MagicMock
) -> None:
    """ChatBot.respond() must append user + assistant dicts, not tuples."""
    from utils import chatbot as chatbot_module
    from utils.chatbot import ChatBot

    with patch("utils.chatbot.os.path.exists", return_value=True), \
         patch("utils.chatbot.Chroma", return_value=mock_vectordb), \
         patch("utils.chatbot.ChatBot.clean_references", return_value="mock references"), \
         patch.object(chatbot_module.APPCFG, "openai_client", mock_openai_client):
        _, history, _ = ChatBot.respond([], "What is RAG?", data_type="Preprocessed doc")

    assert isinstance(history[0], dict), "Expected dict, got tuple — Gradio 5 requires dict format"
    assert isinstance(history[1], dict), "Expected dict, got tuple — Gradio 5 requires dict format"
    assert _is_user_message(history[0], "What is RAG?")
    assert _is_assistant_message(history[1])


def test_chatbot_error_message_is_dict_format() -> None:
    """Error messages appended by ChatBot.respond() must also be dicts."""
    from utils.chatbot import ChatBot

    with patch("utils.chatbot.os.path.exists", return_value=False):
        _, history, _ = ChatBot.respond([], "hi", data_type="Preprocessed doc")

    assert isinstance(history[0], dict), "Error message must be a dict for Gradio 5"
    assert history[0].get("role") == "assistant"


def test_upload_file_rag_mode_appends_dict(mock_vectordb: MagicMock) -> None:
    """UploadFile.process_uploaded_files() must append dicts for RAG mode."""
    from utils import upload_file as upload_module
    from utils.upload_file import UploadFile

    mock_prepare = MagicMock()
    with patch("utils.upload_file.PrepareVectorDB", return_value=mock_prepare):
        _, history = UploadFile.process_uploaded_files(
            files_dir=["/fake/doc.pdf"],
            chatbot=[],
            rag_with_dropdown="Upload doc: Process for RAG",
        )

    assert isinstance(history[0], dict), "UploadFile must append dicts for Gradio 5"
    assert history[0].get("role") == "assistant"


def test_upload_file_dropdown_fallback_appends_dict() -> None:
    """Fallback branch of process_uploaded_files must also append a dict."""
    from utils.upload_file import UploadFile

    _, history = UploadFile.process_uploaded_files(
        files_dir=[],
        chatbot=[],
        rag_with_dropdown="Some other option",
    )

    assert isinstance(history[0], dict), "Fallback message must be a dict for Gradio 5"
    assert history[0].get("role") == "assistant"


def test_memory_string_built_from_dict_history(
    mock_openai_client: MagicMock, mock_vectordb: MagicMock
) -> None:
    """The LLM prompt should incorporate prior dict-format history as readable text."""
    from utils import chatbot as chatbot_module
    from utils.chatbot import ChatBot

    prior_history = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]

    with patch("utils.chatbot.os.path.exists", return_value=True), \
         patch("utils.chatbot.Chroma", return_value=mock_vectordb), \
         patch("utils.chatbot.ChatBot.clean_references", return_value="mock references"), \
         patch.object(chatbot_module.APPCFG, "openai_client", mock_openai_client):
        ChatBot.respond(prior_history, "new question", data_type="Preprocessed doc")

    messages_sent = mock_openai_client.chat.completions.create.call_args[1]["messages"]
    user_prompt = next(m["content"] for m in messages_sent if m["role"] == "user")
    assert "previous question" in user_prompt or "previous answer" in user_prompt, (
        "Memory from dict-format history should appear in the LLM prompt"
    )
