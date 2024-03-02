"""
This module is not part of the main RAG-GPT pipeline and it is only for showing how we can perform RAG using openai and vectordb in the terminal.

To execute the code, after preparing the python environment and the vector database, in the terminal execute:

python src\terminal_q_and_a.py
"""

import openai
import yaml
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from typing import List, Tuple
from utils.load_config import LoadConfig

# For loading openai credentials
APPCFG = LoadConfig()

with open("configs/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

# Load the embedding function
embedding = OpenAIEmbeddings()
# Load the vector database
vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                  embedding_function=embedding)

print("Number of vectors in vectordb:", vectordb._collection.count())

# Prepare the RAG with openai in terminal
while True:
    question = input("\n\nEnter your question or press 'q' to exit: ")
    if question.lower() =='q':
        break
    question = "# user new question:\n" + question
    docs = vectordb.similarity_search(question, k=APPCFG.k)
    retrieved_docs_page_content: List[Tuple] = [
        str(x.page_content)+"\n\n" for x in docs]
    retrived_docs_str = "# Retrieved content:\n\n" + str(retrieved_docs_page_content)
    prompt = retrived_docs_str + "\n\n" + question
    response = openai.ChatCompletion.create(
        engine=APPCFG.llm_engine,
        messages=[
            {"role": "system", "content": APPCFG.llm_system_role},
            {"role": "user", "content": prompt}
        ]
    )
    print(response['choices'][0]['message']['content'])