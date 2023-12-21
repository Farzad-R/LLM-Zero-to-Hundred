from typing import Dict
import inspect
import json
from inspect import Parameter
from pydantic import create_model
from utils.web_search import WebSearch
from typing import List, Dict
import openai


class Apputils:

    @staticmethod
    def jsonschema(f) -> Dict:
        """
        Generate a JSON schema for the input parameters of the given function.

        Parameters:
            f (FunctionType): The function for which to generate the JSON schema.

        Returns:
            Dict: A dictionary containing the function name, description, and parameters schema.
        """
        kw = {n: (o.annotation, ... if o.default == Parameter.empty else o.default)
              for n, o in inspect.signature(f).parameters.items()}
        s = create_model(f'Input for `{f.__name__}`', **kw).schema()
        return dict(name=f.__name__, description=f.__doc__, parameters=s)

    @staticmethod
    def wrap_functions() -> List:
        """
        Wrap several web search functions and generate JSON schemas for each.

        Returns:
            List: A list of dictionaries, each containing the function name, description, and parameters schema.
        """
        return [
            Apputils.jsonschema(WebSearch.retrieve_web_search_results),
            Apputils.jsonschema(WebSearch.web_search_text),
            Apputils.jsonschema(WebSearch.web_search_pdf),
            Apputils.jsonschema(WebSearch.get_instant_web_answer),
            Apputils.jsonschema(WebSearch.web_search_image),
            Apputils.jsonschema(WebSearch.web_search_video),
            Apputils.jsonschema(WebSearch.web_search_news),
            Apputils.jsonschema(WebSearch.web_search_map),
        ]

    @staticmethod
    def execute_json_function(response) -> List:
        """
        Execute a function based on the response from an OpenAI ChatCompletion API call.

        Parameters:
            response: The response object from the OpenAI ChatCompletion API call.

        Returns:
            List: The result of the executed function.
        """
        func_name: str = response.choices[0].message.function_call.name
        func_args: Dict = json.loads(
            response.choices[0].message.function_call.arguments)
        # Call the function with the given arguments
        if func_name == 'retrieve_web_search_results':
            result = WebSearch.retrieve_web_search_results(**func_args)
        elif func_name == 'web_search_text':
            result = WebSearch.web_search_text(**func_args)
        elif func_name == 'web_search_pdf':
            result = WebSearch.web_search_pdf(**func_args)
        elif func_name == 'web_search_image':
            result = WebSearch.web_search_image(**func_args)
        elif func_name == 'web_search_video':
            result = WebSearch.web_search_video(**func_args)
        elif func_name == 'web_search_news':
            result = WebSearch.web_search_news(**func_args)
        elif func_name == 'get_instant_web_answer':
            result = WebSearch.get_instant_web_answer(**func_args)
        elif func_name == 'web_search_map':
            result = WebSearch.web_search_map(**func_args)
        else:
            raise ValueError(f"Function '{func_name}' not found.")
        return result

    @staticmethod
    def ask_llm_function_caller(gpt_model: str, temperature: float, messages: List, function_json_list: List):
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

    @staticmethod
    def ask_llm_chatbot(gpt_model: str, temperature: float, messages: List):
        """
        Generate a response from an OpenAI ChatCompletion API call without specific function calls.

        Parameters:
            gpt_model (str): The name of the GPT model to use.
            temperature (float): The temperature parameter for the API call.
            messages (List): List of message objects for the conversation.

        Returns:
            The response object from the OpenAI ChatCompletion API call.
        """
        response = openai.ChatCompletion.create(
            engine=gpt_model,
            messages=messages,
            temperature=temperature
        )
        return response
