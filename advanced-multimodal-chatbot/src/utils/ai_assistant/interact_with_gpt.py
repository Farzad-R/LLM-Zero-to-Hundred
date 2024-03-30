
import openai
from typing import Dict

def interact_with_gpt_assistant(prompt:str, llm_engine:str, temperature:float, llm_system_role:str) -> Dict:
    """
    Interact with GPT-based chatbot assistant.

    This function sends a prompt to the GPT language model engine and receives a response from it.

    Args:
        prompt (str): The user prompt to send to the chatbot.
        llm_engine (str): The language model engine to use for generating the response.
        temperature (float): The temperature parameter for sampling from the language model.
        llm_system_role (str): The system role content for the language model.

    Returns:
        Dict: A dictionary containing the response from the chatbot.
    """
    response = openai.ChatCompletion.create(
            engine=llm_engine,
            messages=[
                {"role": "system", "content": llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
    return response