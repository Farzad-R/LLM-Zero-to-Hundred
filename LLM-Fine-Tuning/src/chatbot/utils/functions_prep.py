import inspect
from pydantic import create_model
from typing import Dict


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
