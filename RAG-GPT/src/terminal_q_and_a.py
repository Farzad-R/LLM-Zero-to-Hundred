"""
This module is not part of the main RAG-GPT pipeline. It shows how to perform RAG
using OpenAI and ChromaDB in the terminal.

To execute: python src/terminal_q_and_a.py
"""

from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from utils.load_config import LoadConfig

APPCFG = LoadConfig()

embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=APPCFG.persist_directory, embedding_function=embedding)

print("Number of vectors in vectordb:", vectordb._collection.count())

while True:
    question = input("\n\nEnter your question or press 'q' to exit: ")
    if question.lower() == "q":
        break
    question = "# user new question:\n" + question
    docs = vectordb.similarity_search(question, k=APPCFG.k)
    retrieved_docs_page_content: List[Tuple] = [str(x.page_content) + "\n\n" for x in docs]
    retrieved_docs_str = "# Retrieved content:\n\n" + str(retrieved_docs_page_content)
    prompt = retrieved_docs_str + "\n\n" + question
    response = APPCFG.openai_client.chat.completions.create(
        model=APPCFG.llm_engine,
        messages=[
            {"role": "system", "content": APPCFG.llm_system_role},
            {"role": "user", "content": prompt},
        ],
    )
    print(response.choices[0].message.content)
