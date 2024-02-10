import chainlit as cl
import traceback
from chainlit.input_widget import Select
from langchain_utils.load_config import LoadConfig
import os
from dotenv import load_dotenv, find_dotenv
# import yaml
# import pandas as pd
import openai
from pyprojroot import here
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
_ = load_dotenv(find_dotenv())

CFG = LoadConfig()
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(deployment=CFG.embed_model_name,
                              openai_api_key=openai.api_key,
                              openai_api_base=openai.api_base,
                              openai_api_version=openai.api_version,
                              openai_api_type=openai.api_type,
                              # chunk_size=10
                              )
llm = AzureChatOpenAI(
    temperature=CFG.temperature,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_version=openai.api_version,
    openai_api_type=openai.api_type,
    streaming=False,
    deployment_name=CFG.gpt_model)

llm_system_role = """You are a chatbot. You'll receive a prompt that includes retrieved content from the vectorDB based on the user's question, and the source.\
Your task is to respond to the user's new question using the information from the vectorDB without relying on your own knowledge.
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(
        llm_system_role),
    HumanMessagePromptTemplate.from_template("{question}"),
]

prompt = ChatPromptTemplate.from_messages(messages)

chain_type_kwargs = {"prompt": prompt}


@cl.on_chat_start
async def on_chat_start():
    try:
        await cl.Avatar(
            name="chatbot",
            path=str(here("public/langchain.png"))
        ).send()
        # await cl.Avatar(
        #     name="Error",
        #     url=str(here("public/langchain.png"))
        # ).send()
        await cl.Avatar(
            name="User",
            path=str(here("public/me.png"))
        ).send()
        await cl.Message(f"Hello, welcome to `Langchain` Chatbot! How can I help you?").send()
        settings = await cl.ChatSettings(
            [
                Select(id="rag_type",
                       label="Select the retrieval technique",
                       values=[
                           "mmr search", "Similarity search"]
                       ),
                Select(id="splitter_type",
                       label="Select the splitter technique",
                       values=[
                           "Token splitter", "Recursive character splitter"]
                       ),
            ]
        ).send()
        cl.user_session.set(
            "rag_type",
            settings["rag_type"],
        )
        cl.user_session.set("splitter_type", settings["splitter_type"])
    except BaseException as e:
        print(f"Caught error on on_chat_start in app.py: {e}")
        traceback.print_exc()


@cl.on_settings_update
async def setup_agent(settings):
    try:
        user_message = ""
        cl.user_session.set("rag_type", settings["rag_type"])
        if cl.user_session.get("rag_type") is not None:
            user_message += f"{settings['rag_type']} is activated."
        cl.user_session.set("splitter_type", settings["splitter_type"])
        if cl.user_session.get("splitter_type", settings["splitter_type"]) is not None:
            user_message += f" {settings['splitter_type']} is activated."
        await cl.Message(user_message).send()

    except Exception as e:
        await cl.Message("An unexpected error occurred when retrieving the previous sessions. We are looking into it.").send()


@cl.on_message
async def on_message(message: cl.Message):
    try:
        msg = cl.Message(content="")
        await msg.send()
        if cl.user_session.get("splitter_type") == "Recursive character splitter" or cl.user_session.get("splitter_type") == None:
            vectordb = Chroma(persist_directory=str(here(CFG.recursive_vector_db_save_dir)),
                              embedding_function=embeddings)
        elif cl.user_session.get("splitter_type") == "Token splitter":
            vectordb = Chroma(persist_directory=str(here(CFG.token_vector_db_save_dir)),
                              embedding_function=embeddings)
        # default value: mmr
        if cl.user_session.get("rag_type") == "mmr search" or cl.user_session.get("rag_type") == None:
            print("RAG-Type: mmr search\n")
            retriever = vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": CFG.langchain_k},
            )

        elif cl.user_session.get("rag_type") == "Similarity search":
            print("RAG-Type: Similarity search\n")
            retriever = vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": CFG.langchain_k},
            )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        result = chain(message.content)
        answer = result["answer"]
        await cl.Message(str(answer)).send()
    except BaseException as e:
        print(f"Caught error on on_message in app.py: {e}")
        traceback.print_exc()
        await cl.Message("An error occured while processing your query. Please try again later.").send()
