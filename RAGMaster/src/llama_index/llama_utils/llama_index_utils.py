from pyprojroot import here
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index import (load_index_from_storage,
                         ServiceContext,
                         StorageContext,
                         SimpleDirectoryReader,
                         VectorStoreIndex,
                         )
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import (SentenceTransformerRerank,
                                               MetadataReplacementPostProcessor)
from llama_index.node_parser import get_leaf_nodes, HierarchicalNodeParser
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


def load_llm_and_embedding_models(gpt_model: str = "gpt-35-turbo-16k", embed_model_name: str = "bge-small-en-v1.5"):
    # Load the GPT model and the embedding model from AzureOpenAI
    llm = AzureOpenAI(
        engine=gpt_model,
        model=gpt_model,
        deployment_name=os.getenv("gpt_deployment_name"),
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )
    if embed_model_name == "text-embedding-ada-002":
        embed_model = AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name=os.getenv("embed_deployment_name"),
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )
    elif embed_model_name == "bge-small-en-v1.5":
        embed_model = "local:BAAI/bge-small-en-v1.5"
    return llm, embed_model


def set_service_context(llm, embed_model):
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    return service_context


def load_documents(documents_dir):
    documents = SimpleDirectoryReader(
        input_files=[here(f"{documents_dir}/{d}")
                     for d in os.listdir(here(documents_dir))]  # gets each file and create the full path
    ).load_data()
    return documents


def build_sentence_window_index(document, llm, save_dir, embed_model="local:BAAI/bge-small-en-v1.5", window_size: int = 3):
    """
    Builds an index of sentence windows from a given document using a specified language model and embedding model.

    This function creates a sentence window node parser with default settings and uses it to parse the document.
    It then initializes a service context with the provided language model and embedding model. If the save directory
    does not exist, it creates a new VectorStoreIndex from the document and persists it to the specified directory.
    If the save directory already exists, it loads the index from storage.

    Args:
        document (str): The text document to be indexed.
        llm (LanguageModel): The language model to be used for parsing and embedding.
        embed_model (str, optional): The identifier for the embedding model to be used. Defaults to "local:BAAI/bge-small-en-v1.5".
        save_dir (str, optional): The directory where the sentence index will be saved or loaded from. Defaults to "sentence_index".

    Returns:
        VectorStoreIndex: The index of sentence windows created or loaded from the save directory.

    Raises:
        OSError: If there is an issue with creating or accessing the save directory.
        Other exceptions may be raised by the underlying storage or indexing operations.
    """
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document], service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def build_automerging_index(
    documents,
    llm,
    save_dir,
    embed_model="local:BAAI/bge-small-en-v1.5",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_sentence_window_query_engine(
    sentence_index,
    rerank_model: str = "BAAI/bge-reranker-base",
    similarity_top_k: int = 6,
    rerank_top_n: int = 2,
):
    """
    Initializes a query engine for sentence window indexing with postprocessing capabilities.

    This function sets up a query engine using a given sentence index. It defines postprocessors for metadata replacement
    and reranking based on sentence embeddings. The query engine is configured to return a specified number of top
    similar results, and to rerank a subset of those results using a sentence transformer model.

    Args:
        sentence_index (VectorStoreIndex): The index of sentence windows to be queried.
        similarity_top_k (int, optional): The number of top similar results to return from the initial query.
                                          Defaults to 6.
        rerank_top_n (int, optional): The number of top results to rerank using the sentence transformer model.
                                      Defaults to 2.

    Returns:
        QueryEngine: The query engine configured for sentence window indexing with postprocessing.

    Raises:
        ValueError: If the provided `similarity_top_k` or `rerank_top_n` are not valid integers or are out of expected range.
        Other exceptions may be raised by the underlying query engine or postprocessing operations.
    """
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model=rerank_model
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[
            postproc, rerank]
    )
    return sentence_window_engine


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k: int = 12,
    rerank_top_n: int = 2,
):
    base_retriever = automerging_index.as_retriever(
        similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine
