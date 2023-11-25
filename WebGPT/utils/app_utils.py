
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
        kw = {n: (o.annotation, ... if o.default == Parameter.empty else o.default)
              for n, o in inspect.signature(f).parameters.items()}
        s = create_model(f'Input for `{f.__name__}`', **kw).schema()
        return dict(name=f.__name__, description=f.__doc__, parameters=s)

    @staticmethod
    def wrap_functions() -> List:
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
        response = openai.ChatCompletion.create(
            engine=gpt_model,
            messages=messages,
            temperature=temperature
        )
        return response
