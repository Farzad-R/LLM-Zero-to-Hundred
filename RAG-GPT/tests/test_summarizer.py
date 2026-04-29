import pytest
from unittest.mock import patch, MagicMock, call


def _make_page(content: str) -> MagicMock:
    page = MagicMock()
    page.page_content = content
    return page


def test_get_llm_response_calls_completions_create(mock_openai_client: MagicMock) -> None:
    """get_llm_response() calls client.chat.completions.create exactly once."""
    from utils import summarizer as summarizer_module
    from utils.summarizer import Summarizer

    with patch.object(summarizer_module.APPCFG, "openai_client", mock_openai_client):
        result = Summarizer.get_llm_response(
            gpt_model="gpt-4o-mini",
            temperature=0.0,
            llm_system_role="You are a summarizer.",
            prompt="Summarize this."
        )

    mock_openai_client.chat.completions.create.assert_called_once()
    assert result == "Mock LLM response"


def test_get_llm_response_passes_model_param(mock_openai_client: MagicMock) -> None:
    """The model parameter is forwarded to the OpenAI client."""
    from utils import summarizer as summarizer_module
    from utils.summarizer import Summarizer

    with patch.object(summarizer_module.APPCFG, "openai_client", mock_openai_client):
        Summarizer.get_llm_response("gpt-4o-mini", 0.0, "sys", "prompt")

    kwargs = mock_openai_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "gpt-4o-mini"


def test_summarize_single_page_doc(mock_openai_client: MagicMock, tmp_path) -> None:
    """A single-page PDF uses its content directly, then calls get_llm_response once."""
    from utils import summarizer as summarizer_module
    from utils.summarizer import Summarizer

    single_page = [_make_page("Page one content")]

    with patch("utils.summarizer.PyPDFLoader") as mock_loader, \
         patch.object(summarizer_module.APPCFG, "openai_client", mock_openai_client):
        mock_loader.return_value.load.return_value = single_page
        result = Summarizer.summarize_the_pdf(
            file_dir="fake.pdf",
            max_final_token=3000,
            token_threshold=0,
            gpt_model="gpt-4o-mini",
            temperature=0.0,
            summarizer_llm_system_role="Summarize within {} tokens.",
            final_summarizer_llm_system_role="Give a final summary.",
            character_overlap=100,
        )

    # Single page: only the final summary LLM call is made
    assert mock_openai_client.chat.completions.create.call_count == 1
    assert result == "Mock LLM response"
