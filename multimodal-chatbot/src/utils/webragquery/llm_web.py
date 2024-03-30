
from typing import List, Dict
import openai


class LLMWeb:
    @staticmethod
    def prepare_messages(search_result: List, user_query: str, llm_system_role: str, input_chat_history: str) -> List[Dict]:
        """
        Prepares a list of messages with roles and content based on web search results and a user query.

        This function formats the user's query andS the web search results into a structured list of dictionaries,
        where each dictionary represents a message with a specified role (either 'system' or 'user') and its content.
        The 'system' message contains the role of the LLM (Language Learning Model) system, and the 'user' message
        contains the formatted user query along with the web search results.

        Parameters:
            search_result (List): A list containing the results of a web search.
            user_query (str): The user's query as a string.
            llm_system_role (str): The role of the LLM system to be included in the system message.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary has two keys: 'role' and 'content'.
                    The 'role' key denotes the role of the message ('system' or 'user'), and the 'content'
                    key contains the message content.
        """
        query = f"{input_chat_history}\n\n, # Web search results:\n\n{search_result}\n\n, # User's new query: {user_query}"
        messages = [
            {"role": "system", "content": llm_system_role},
            {"role": "user", "content": query}
        ]
        return messages

    @staticmethod
    def ask(gpt_model: str, temperature: float, messages: List) -> str:
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
        return response["choices"][0]["message"]["content"]
