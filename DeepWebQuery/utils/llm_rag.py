from typing import List, Dict, Tuple
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


class LLM_RAG:
    @staticmethod
    def search_vectordb(persist_directory: str, user_query: str, k: int = 3):
        embedding_model = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory=persist_directory,
                          embedding_function=embedding_model)
        docs = vectordb.similarity_search(user_query, k=k)
        retrieved_docs_page_content = [
            str(x.page_content)+"\n\n" for x in docs]
        retrieved_docs_page_content = "# Retrieved content:\n" + \
            str(retrieved_docs_page_content)
        return retrieved_docs_page_content

    @staticmethod
    def prepare_messages(persist_directory: str, user_query: str, llm_system_role: str, input_chat_history: str, k: int = 3) -> List[Dict]:
        retrieved_docs_page_content = LLM_RAG.search_vectordb(
            persist_directory=persist_directory, user_query=user_query)
        query = f"# Chat history: {input_chat_history}\n\n, # User's new query: {user_query}\n\n, # vector search result on the url:\n\n, {retrieved_docs_page_content}"
        messages = [
            {"role": "system", "content": llm_system_role},
            {"role": "user", "content": input_chat_history + query}
        ]
        return messages

    @staticmethod
    def ask(gpt_model: str, temperature: float, messages: List):
        """
        Generate a response from an OpenAI ChatCompletion API call without specific function calls.

        Parameters:
            - gpt_model (str): The name of the GPT model to use.
            - temperature (float): The temperature parameter for the API call.
            - messages (List): List of message objects for the conversation.

        Returns:
            The response object from the OpenAI ChatCompletion API call.
        """
        response = openai.ChatCompletion.create(
            engine=gpt_model,
            messages=messages,
            temperature=temperature
        )
        return response
