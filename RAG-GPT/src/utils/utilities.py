import tiktoken


def count_num_tokens(text: str, model: str) -> int:
    """
    Returns the number of tokens in the given text.
    Args:
        text (str): The text to count tokens in.
        model (str, optional): The name of the GPT model to use. Defaults to the model specified in the app config.

    Returns:
        int: The number of tokens in the text.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
