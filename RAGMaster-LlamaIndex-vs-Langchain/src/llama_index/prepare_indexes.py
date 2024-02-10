# Load the libraries
from llama_index import (set_global_service_context,
                         VectorStoreIndex,
                         Document)

from llama_utils.llama_index_utils import (load_llm_and_embedding_models,
                                           set_service_context,
                                           load_documents,
                                           build_sentence_window_index,
                                           build_automerging_index)
from llama_utils.load_config import LoadConfig
from pyprojroot import here
CFG = LoadConfig()


def prep_llama_indexes():
    # Load LLM and embedding model
    llm, embed_model = load_llm_and_embedding_models(
        gpt_model=CFG.gpt_model, embed_model_name=CFG.embed_model_name)
    # Set the serivce context
    service_context = set_service_context(
        llm=llm,
        embed_model=embed_model,
    )
    set_global_service_context(service_context)

    # Load the documents
    page_separated_documents = load_documents(
        CFG.documents_dir)
    print("Documents are loaded.")

    # =========================
    # Prepare Basic RAG index
    # =========================
    # Processing for `Basic RAG`: Turn all the documents in one long piece of text
    print("Processing the documents and creating the index for Basic RAG...")
    merged_document = Document(
        text="\n\n".join([doc.text for doc in page_separated_documents]))
    # Generate the index
    index = VectorStoreIndex.from_documents([merged_document],
                                            service_context=service_context)
    # Save the index
    index.storage_context.persist(here(CFG.basic_rag_index_save_dir))
    print(f"Index of Basic RAG is saved in {CFG.basic_rag_index_save_dir}.\n")

    # ============================
    # Prepare Page wise RAG index
    # ============================
    print("Processing the documents and creating the index for Page-wise RAG...")
    index = VectorStoreIndex.from_documents(page_separated_documents)
    # Save the index
    index.storage_context.persist(here(CFG.pagewise_rag_index_save_dir))
    print(
        f"Index of Page-wise RAG is saved in {CFG.pagewise_rag_index_save_dir}.\n")
    # ==================================
    # Prepare Sentence retrieval index
    # ==================================
    print("Processing the documents and creating the index for Sentence Retrieval...")
    sentence_index = build_sentence_window_index(
        merged_document,
        llm,
        embed_model=embed_model,
        save_dir=here(CFG.sentence_index_save_dir),
        window_size=CFG.sentence_window_size
    )
    print(
        f"Index of Sentence Retrieval is saved in {CFG.sentence_index_save_dir}.\n")
    # ======================================
    # Prepare Auto-merging retrieval index
    # ======================================
    print("Processing the documents and creating the index for Auto-merging Retrieval...")
    automerging_index = build_automerging_index(
        page_separated_documents,
        llm,
        embed_model=embed_model,
        save_dir=here(CFG.auto_merging_retrieval_index_save_dir),
        chunk_sizes=CFG.chunk_sizes
    )
    print(
        f"Index of Auto-merging Retrieval is saved in {CFG.auto_merging_retrieval_index_save_dir}.")


if __name__ == "__main__":
    prep_llama_indexes()
