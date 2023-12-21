from typing import List
from utils.cfg import LoadConfig
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import time
import openai
APPCFG = LoadConfig()


class PrepDocForFullSummary:
    def __init__(
            self,
            data_directory: str,
            chunk_size: int = 1000,
            chunk_overlap: int = 250,
            # max_token: int = 8192
    ) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        """Other options: CharacterTextSplitter, NotionDirectoryLoader, TokenTextSplitter, etc."""
        # self.data_directory = data_directory
        self.doc = self._load_the_doc(data_directory)
        self.chunked_doc = self._chunk_documents(self.doc)
        # max_chunk_summary_length = (max_token/len(self.chunked_doc)) - 100

    def _load_the_doc(self) -> List:
        """
        Load the requested document from the specified directory.

        Returns:
            List: A list with the content of the loaded document.
        """
        docs = []
        docs.extend(PyPDFLoader(
            self.data_directory[0])).load()
        print("Number of pages:", len(docs), "\n\n")
        return docs

    def _chunk_documents(self, docs: List) -> List:
        """
        Chunk the loaded document using the specified text splitter.

        Parameters:
            docs (List): The list of loaded document.

        Returns:
            List: A list of chunked documents.

        """
        print("Chunking documents...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks:", len(chunked_documents), "\n\n")
        return chunked_documents


def summarize(files_dir: List, chatbot: List):
    print("=========================================")
    print("=========================================")
    print(files_dir, "\n")
    prep_doc_instance = PrepDocForFullSummary(data_directory=files_dir)
    full_summary_list = []
    for chunk in prep_doc_instance.chunked_doc:
        print(chunk, "\n")
        last_summary = full_summary_list[-1] if len(
            full_summary_list) > 0 else " "
        print(last_summary, "\n")
        prompt = f"# Last chunk summary {last_summary}\n\n# New chunk to be summarized {chunk}"
        response = openai.ChatCompletion.create(
            engine=APPCFG.llm_engine,
            messages=[
                {"role": "system", "content": "Summarize the new chunks in less than 200 words. You will receive the previous summary of the document just for more info."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            # stream=False
        )
        full_summary_list.append(
            response["choices"][0]["message"]["content"])
        time.sleep(2)  # For not hitting the max request per minute
    print(full_summary_list, "\n")

    full_summary = ""
    for i in full_summary_list:
        full_summary += i
    chatbot.append(
        (" ", response["choices"][0]["message"]["content"]))
    print(full_summary)
    print("=========================================")
    print("=========================================")
    return "", chatbot
