import chainlit as cl
import traceback
from chainlit.input_widget import Select
from pyprojroot import here
from llama_index import (load_index_from_storage,
                         StorageContext)
from llama_utils.llama_index_utils import (get_sentence_window_query_engine,
                                           get_automerging_query_engine,
                                           load_llm_and_embedding_models,
                                           set_service_context)
from llama_index import set_global_service_context
from llama_utils.load_config import LoadConfig
CFG = LoadConfig()

llm, embed_model = load_llm_and_embedding_models(
    gpt_model=CFG.gpt_model, embed_model_name=CFG.embed_model_name)
service_context = set_service_context(
    llm=llm,
    embed_model=embed_model,
)
set_global_service_context(service_context)


@cl.on_chat_start
async def on_chat_start():
    try:
        await cl.Avatar(
            name="Enterprise LLM",
            path=str(here("public/llama-index.png"))
        ).send()
        await cl.Avatar(
            name="Error",
            url=str(here("public/llama-index.png"))
        ).send()
        await cl.Avatar(
            name="User",
            path=str(here("public/me.png"))
        ).send()
        await cl.Message(f"Hello, welcome to `LlamaIndex` Chatbot! How can I help you?").send()
        settings = await cl.ChatSettings(
            [
                Select(id="rag_type",
                       label="Select the RAG technique",
                       values=[
                           "page-wise RAG", "Basic RAG", "Sentence Retrieval", "Auto-merging retrieval"]
                       ),
            ]
        ).send()
        cl.user_session.set(
            "rag_type",
            settings["rag_type"],
        )
    except BaseException as e:
        print(f"Caught error on on_chat_start in app.py: {e}")
        traceback.print_exc()


@cl.on_settings_update
async def setup_agent(settings):
    try:
        cl.user_session.set("rag_type", settings["rag_type"])
        await cl.Message(f"{settings['rag_type']} is activated.").send()
    except Exception as e:
        await cl.Message("An unexpected error occurred when retrieving the previous sessions. We are looking into it.").send()


@cl.on_message
async def on_message(message: cl.Message):
    try:
        msg = cl.Message(content="")
        await msg.send()
        if cl.user_session.get("rag_type") == "page-wise RAG" or cl.user_session.get("rag_type") == None:

            storage_context = StorageContext.from_defaults(
                persist_dir=CFG.pagewise_rag_index_save_dir)
            index = load_index_from_storage(storage_context)
            query_engine = index.as_query_engine()
            response = query_engine.query(
                message.content
            )
        elif cl.user_session.get("rag_type") == "Basic RAG":
            print("RAG-Type: Basic RAG\n")
            storage_context = StorageContext.from_defaults(
                persist_dir=CFG.basic_rag_index_save_dir)
            index = load_index_from_storage(storage_context)
            query_engine = index.as_query_engine()
            response = query_engine.query(
                message.content
            )
        elif cl.user_session.get("rag_type") == "Sentence Retrieval":
            print("RAG-Type: Sentence Retrieval\n")
            storage_context = StorageContext.from_defaults(
                persist_dir=CFG.sentence_index_save_dir)
            sentence_index = load_index_from_storage(storage_context)
            sentence_window_engine = get_sentence_window_query_engine(
                sentence_index=sentence_index,
                rerank_model=CFG.rerank_model,
                similarity_top_k=CFG.sentence_retrieval_similarity_top_k,
                rerank_top_n=CFG.sentence_retrieval_rerank_top_n)
            response = sentence_window_engine.query(
                message.content
            )
        else:  # cl.user_session.get("rag_type") == Auto-merging retrieval
            # Rebuild storage context
            print("RAG-Type: Auto-merging retrieval\n")
            storage_context = StorageContext.from_defaults(
                persist_dir=CFG.auto_merging_retrieval_index_save_dir)
            # Load index from the storage context
            automerging_index = load_index_from_storage(storage_context)
            automerging_query_engine = get_automerging_query_engine(
                automerging_index,
                similarity_top_k=CFG.auto_merging_retrieval_similarity_top_k,
                rerank_top_n=CFG.auto_merging_retrieval_rerank_top_n,
            )
            response = automerging_query_engine.query(
                message.content
            )
        await cl.Message(str(response)).send()
    except BaseException as e:
        print(f"Caught error on on_message in app.py: {e}")
        traceback.print_exc()
        await cl.Message("An error occured while processing your query. Please try again later.").send()
