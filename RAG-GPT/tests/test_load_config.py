import pytest
from unittest.mock import patch, MagicMock
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings


@pytest.fixture
def cfg():
    from utils.load_config import LoadConfig
    with patch.object(LoadConfig, "create_directory"), \
         patch.object(LoadConfig, "remove_directory"):
        return LoadConfig()


def test_llm_engine_is_gpt4o_mini(cfg) -> None:
    assert cfg.llm_engine == "gpt-4o-mini"


def test_openai_client_is_openai_instance(cfg) -> None:
    assert isinstance(cfg.openai_client, OpenAI)


def test_no_azure_config_method(cfg) -> None:
    assert not hasattr(cfg, "load_openai_cfg"), (
        "load_openai_cfg should be removed after Azure migration"
    )


def test_embedding_model_is_openai_embeddings(cfg) -> None:
    assert isinstance(cfg.embedding_model, OpenAIEmbeddings)


def test_retrieval_k_loaded(cfg) -> None:
    assert isinstance(cfg.k, int) and cfg.k > 0


def test_chunk_config_loaded(cfg) -> None:
    assert isinstance(cfg.chunk_size, int) and cfg.chunk_size > 0
    assert isinstance(cfg.chunk_overlap, int) and cfg.chunk_overlap >= 0


def test_memory_pairs_loaded(cfg) -> None:
    assert isinstance(cfg.number_of_q_a_pairs, int) and cfg.number_of_q_a_pairs > 0
