from llama_index.indices.postprocessor import (SentenceTransformerRerank,
                                               MetadataReplacementPostProcessor)
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import AutoMergingRetriever


def get_sentence_window_query_engine(
    sentence_index,
    rerank_model,
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
    rerank_model,
    similarity_top_k: int = 12,
    rerank_top_n: int = 2,
):
    base_retriever = automerging_index.as_retriever(
        similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model=rerank_model
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine
