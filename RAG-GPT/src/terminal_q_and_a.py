import openai
import yaml
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from typing import List, Tuple
from utils.cfg import LoadConfig
from pyprojroot import here
# For loading openai credentials
APPCFG = LoadConfig()

with open("configs/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)
# LLM configs
llm_engine = app_config["llm_config"]["engine"]
llm_system_role = app_config["llm_config"]["llm_system_role"]
llm_temperature = app_config["llm_config"]["temperature"]

# directories
persist_directory = str(here(app_config["directories"]["persist_directory"]))

# Retrieval configs
k = app_config["retrieval_config"]["k"]


embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

print("Number of vectors in vectordb:", vectordb._collection.count())
while True:
    question = input("\n\nEnter your question or press 'q' to exit: ")
    if question.lower() == 'q':
        break
    question = "# User new question:\n" + question
    docs = vectordb.similarity_search(question, k=k)
    retrieved_docs_page_content: List[Tuple] = [
        str(x.page_content)+"\n\n" for x in docs]
    retrieved_docs_str = "# Retrieved content:\n\n" + \
        str(retrieved_docs_page_content)
    prompt = retrieved_docs_str + "\n\n" + question
    response = openai.ChatCompletion.create(
        engine=llm_engine,
        messages=[
            {"role": "system", "content": llm_system_role},
            {"role": "user", "content": prompt}
        ],
        temperature=llm_temperature,
        stream=False
    )

    print(response["choices"][0]["message"]["content"])
