# https://medium.com/@50_shades_of_gradient/retrieval-augmented-generation-citations-rag-c-with-cohere-langchain-qdrant-157f9b4cce22
# https://python.langchain.com/docs/integrations/llms/azure_openai
from dotenv import load_dotenv, find_dotenv
import yaml
import os
import time
import pandas as pd
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
start_time = time.time()
_ = load_dotenv(find_dotenv())
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")

with open(here("configs/config.yml")) as cfg:
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)

splitter_type = cfg["langchain_cfg"]["splitter_type"]

embeddings = OpenAIEmbeddings(deployment=cfg["llm_cfg"]["embed_model_name"],
                              openai_api_key=openai.api_key,
                              openai_api_base=openai.api_base,
                              openai_api_version=openai.api_version,
                              openai_api_type=openai.api_type,
                              # chunk_size=10
                              )
llm = AzureChatOpenAI(
    temperature=cfg["llm_cfg"]["temperature"],
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_version=openai.api_version,
    openai_api_type=openai.api_type,
    streaming=False,
    deployment_name=cfg["llm_cfg"]["gpt_model"])

if splitter_type == "recursive":
    vectordb = Chroma(persist_directory=str(here(cfg["langchain_cfg"]["recursive_vector_db_save_dir"])),
                      embedding_function=embeddings)
elif splitter_type == "token":
    vectordb = Chroma(persist_directory=str(here(cfg["langchain_cfg"]["token_vector_db_save_dir"])),
                      embedding_function=embeddings)

"""search_type:

- "similarity": uses similarity search in the retriever object where it 
selects text chunk vectors that are most similar to the question vector. 

- "mmr": uses the maximum marginal relevance search where it optimizes for 
similarity to query AND diversity among selected documents."""
search_type = cfg["langchain_cfg"]["search_type"]
retriever = vectordb.as_retriever(
    search_type=search_type,  # Also test "similarity"
    search_kwargs={"k": cfg["langchain_cfg"]["k"]},
)


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

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
# test:
# question = "Explain is the architecture of vision transformer model"
# result = chain(question)
# terminal_message = f"Question {1}:\n{question}\n\nAnswer:\n{result['answer']}\n\n\n"
# print(terminal_message)
# print()


questions_df = pd.read_excel(os.path.join(
    here(cfg["eval_questions_dir"]), cfg["eval_file_name"]))
print("\n\n")
print(
    f"processing the questions for langchain with {splitter_type} splitter_type and `{search_type}` search_type.")
print("--------------------------------------\n\n")
for idx, row in questions_df.iterrows():
    inference_start_time = time.time()

    question = row["question"]
    result = chain(question)
    inference_end_time = time.time()
    infernece_time = time.time() - inference_start_time
    answer = result["answer"]
    column_name = f"langchain_{splitter_type}_{search_type}_result"
    questions_df.at[idx, column_name] = answer
    time_column_name = f"langchain_{splitter_type}_{search_type}_inference_time"
    questions_df.at[idx, time_column_name] = round(infernece_time, 2)
    terminal_message = f"Question {idx}:\n{question}\n\nAnswer:\n{result['answer']}"
    print(terminal_message)
    print(f"Inference time: {infernece_time}")
    print("--------------------------------------\n\n")

questions_df.to_excel(
    os.path.join(here(cfg["eval_questions_dir"]), cfg["eval_file_name"]), index=False)

print(
    f"Execution was successfull and {cfg['eval_file_name']} file is saved.\n\n")
end_time = time.time()
# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution Time: {round(execution_time, 2)} seconds")
