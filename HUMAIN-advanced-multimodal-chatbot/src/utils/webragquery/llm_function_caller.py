import openai
from typing import List, Dict


class LLMFuntionCaller:
    @staticmethod
    def prepare_messages(llm_function_caller_system_role: str, input_chat_history: str, user_query: str) -> List[Dict]:
        """
        Prepares a list of message dictionaries to be used by a language model.

        This method formats the system role information and input chat history, along with the user's message,
        into a structured list of dictionaries. Each dictionary contains a 'role' key to denote whether the
        message is from the 'system' or the 'user', and a 'content' key with the actual message content.

        Parameters:
            llm_function_caller_system_role (str): A string representing the system's role or state information.
            input_chat_history (str): The chat history that the system should consider before generating a response.
            message (str): The latest message from the user that needs to be processed.

        Returns:
            List[Dict]: A list containing two dictionaries, one for the system and one for the user, each with
                         'role' and 'content' keys.
        """
        query = f"{input_chat_history}\n\n, # User's new query: {user_query}"
        return [
            {"role": "system", "content": str(
                llm_function_caller_system_role)},
            {"role": "user", "content": query}
        ]

    @staticmethod
    def ask(gpt_model: str, temperature: float, messages: List, function_json_list: List):
        """
        Generate a response from an OpenAI ChatCompletion API call with specific function calls.

        Parameters:
            gpt_model (str): The name of the GPT model to use.
            temperature (float): The temperature parameter for the API call.
            messages (List): List of message objects for the conversation.
            function_json_list (List): List of function JSON schemas.

        Returns:
            The response object from the OpenAI ChatCompletion API call.
        """
        response = openai.ChatCompletion.create(
            engine=gpt_model,
            messages=messages,
            functions=function_json_list,
            function_call="auto",
            temperature=temperature
        )
        return response
