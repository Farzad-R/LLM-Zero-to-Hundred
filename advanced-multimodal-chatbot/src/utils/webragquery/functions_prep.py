import inspect
import json
from pydantic import create_model
from typing import List, Dict
from utils.webragquery.specific_url_prep_func import prepare_the_requested_url_for_q_and_a
from utils.webragquery.web_search_funcs import WebSearch
from utils.webragquery.web_summarizer import WebSummarizer


class PrepareFunctions:
    @staticmethod
    def jsonschema(f) -> Dict:
        """
        Generate a JSON schema for the input parameters of the given function.

        Parameters:
            f (FunctionType): The function for which to generate the JSON schema.

        Returns:
            Dict: A dictionary containing the function name, description, and parameters schema.
        """
        kw = {n: (o.annotation, ... if o.default == inspect.Parameter.empty else o.default)
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
            PrepareFunctions.jsonschema(
                WebSummarizer.summarize_the_webpage),  # webpage summarization functionality
            PrepareFunctions.jsonschema(WebSearch.retrieve_web_search_results),
            PrepareFunctions.jsonschema(WebSearch.get_instant_web_answer),
            PrepareFunctions.jsonschema(WebSearch.web_search_pdf),
            PrepareFunctions.jsonschema(WebSearch.web_search_video),
            PrepareFunctions.jsonschema(WebSearch.web_search_news),
            # PrepareFunctions.jsonschema(WebSearch.web_search_text),
            PrepareFunctions.jsonschema(
                prepare_the_requested_url_for_q_and_a)  # rag functionality
        ]

    @staticmethod
    def execute_json_function(response) -> List | bool:
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
        if func_name == 'summarize_the_webpage':
            result = WebSummarizer.summarize_the_webpage(**func_args)
        elif func_name == 'retrieve_web_search_results':
            result = WebSearch.retrieve_web_search_results(**func_args)
        elif func_name == 'web_search_news':
            result = WebSearch.web_search_news(**func_args)
        elif func_name == 'get_instant_web_answer':
            result = WebSearch.get_instant_web_answer(**func_args)
        elif func_name == 'web_search_video':
            result = WebSearch.web_search_video(**func_args)
        elif func_name == 'web_search_pdf':
            result = WebSearch.web_search_pdf(**func_args)
        elif func_name == 'web_search_text':
            result = WebSearch.web_search_text(**func_args)
        elif func_name == 'prepare_the_requested_url_for_q_and_a':
            result = prepare_the_requested_url_for_q_and_a(**func_args)
        else:
            raise ValueError(f"Function '{func_name}' not found.")
        return result
