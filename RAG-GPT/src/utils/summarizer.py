
from langchain.document_loaders import PyPDFLoader
from utils.utilities import count_num_tokens
import openai


class Summarizer:
    """
    A class for summarizing PDF documents using OpenAI's ChatGPT engine.

    Attributes:
        None

    Methods:
        summarize_the_pdf:
            Summarizes the content of a PDF file using OpenAI's ChatGPT engine.

        get_llm_response:
            Retrieves the response from the ChatGPT engine for a given prompt.

    Note: Ensure that you have the required dependencies installed and configured, including the OpenAI API key.
    """
    @staticmethod
    def summarize_the_pdf(
        file_dir: str,
        max_final_token: int,
        token_threshold: int,
        gpt_model: str,
        temperature: float,
        summarizer_llm_system_role: str,
    ):
        """
        Summarizes the content of a PDF file using OpenAI's ChatGPT engine.

        Args:
            file_dir (str): The path to the PDF file.
            max_final_token (int): The maximum number of tokens in the final summary.
            token_threshold (int): The threshold for token count reduction.
            gpt_model (str): The ChatGPT engine model name.
            temperature (float): The temperature parameter for ChatGPT response generation.
            summarizer_llm_system_role (str): The system role for the summarizer.

        Returns:
            str: The final summarized content.
        """
        docs = []
        docs.extend(PyPDFLoader(file_dir).load())
        print(f"Document length: {len(docs)}")
        max_summarizer_output_token = int(
            max_final_token/len(docs)) - token_threshold
        full_summary = ""
        counter = 1
        print("Generating the summary..")
        for i in range(len(docs)):
            full_summary += Summarizer.get_llm_response(
                gpt_model,
                temperature,
                summarizer_llm_system_role,
                max_summarizer_output_token,
                prompt=docs[i].page_content
            )
            print(f"Page {counter} was summarized. ", end="")
            counter += 1
        print("\nFull summary token length:", count_num_tokens(
            full_summary, model=gpt_model))
        final_summary = Summarizer.get_llm_response(
            gpt_model,
            temperature,
            summarizer_llm_system_role,
            max_summarizer_output_token,
            prompt=full_summary
        )
        return final_summary

    @staticmethod
    def get_llm_response(gpt_model: str, temperature: float, summarizer_llm_system_role: str, max_summarizer_output_token: int, prompt: str):
        """
        Retrieves the response from the ChatGPT engine for a given prompt.

        Args:
            gpt_model (str): The ChatGPT engine model name.
            temperature (float): The temperature parameter for ChatGPT response generation.
            summarizer_llm_system_role (str): The system role for the summarizer.
            max_summarizer_output_token (int): The maximum number of tokens for the summarizer output.
            prompt (str): The input prompt for the ChatGPT engine.

        Returns:
            str: The response content from the ChatGPT engine.
        """
        response = openai.ChatCompletion.create(
            engine=gpt_model,
            messages=[
                {"role": "system", "content": summarizer_llm_system_role.format(
                    max_summarizer_output_token)},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
