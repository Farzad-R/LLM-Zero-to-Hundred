import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from typing import List
import yaml
from utils.app_utils import Apputils
from utils.prepare_url_vectordb import PrepareURLVectorDB


class URLPrep:
    """
    A class to check URLs within a given text for their content type.

    Attributes:
        text (str): The text containing URLs to be checked.
    """

    def __init__(self, text: str) -> None:
        """
        Initializes the URLCheck instance with the provided text.

        Parameters:
            text (str): The text containing URLs to be checked.
        """
        self.text = text
        with open("configs/app_config.yml") as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
            self.chunk_size = app_config["RAG"]["chunk_size"]
            self.chunk_overlap = app_config["RAG"]["chunk_overlap"]
            self.persist_directory = app_config["RAG"]["persist_directory"]
            # Let's remove the vectorDB if it already exists from the previous url.
            Apputils.remove_directory(self.persist_directory)
            # create a new one for the new url.
            Apputils.create_directory(self.persist_directory)
            self.embedding_model_engine = app_config["RAG"]["embedding_model_engine"]

    def _extract_urls(self, text: str):
        """
        Extracts URLs from the provided text using a regular expression.
        Can handle multiple URLs with or without http/https scheme.

        Parameters:
            text (str): The text from which to extract URLs.

        Returns:
            List[str]: A list of extracted URLs.
        """
        # Regex pattern to extract URLs (http, https, and www)
        # Use non-capturing groups with (?:...) to avoid returning tuples
        url_pattern = r"(?:https?://www\.|www\.|https?://)[a-zA-Z0-9\.-]+\.[a-zA-Z]{2,}(?:/\S*)?"
        urls = re.findall(url_pattern, text)
        # Prepend 'http://' if 'www.' is found without 'http://' or 'https://'
        urls = ['http://' + url if not url.startswith(
            'http') and url.startswith('www.') else url for url in urls]
        return urls

    def _identify_content_type(self, url: str) -> str:
        """
        Identifies the content type of the given URL by making an HTTP request and analyzing the content.
        NOTE: ___Can handle both youtube links and webpages___
        Parameters:
            url (str): The URL whose content type is to be identified.

        Returns:
            str: A string describing the content type or an error message.
        """
        try:
            response = requests.get(url)
            if response.status_code == 200:
                html_content = response.text
                soup = BeautifulSoup(html_content, "html.parser")

                # Look for article tags or other common structures
                if soup.find("article") or soup.find("main") or soup.find(attrs={"role": "article"}):
                    return "Webpage article"
                elif "youtube.com" in urlparse(url).netloc or "youtu.be" in urlparse(url).netloc:
                    return "YouTube video"
                else:
                    return "Unknown content type"
            else:
                return "Error: Unable to access content"
        except requests.RequestException as e:
            return f'Error: {e}'

    def prepare_vector_db(self):
        """NOTE: ___Cannot handle both youtube links and webpages or multiple urls for now___"""
        urls = self._extract_urls(self.text)
        url_status = self._identify_content_type(urls[0])
        if url_status == "Webpage article":
            prepare_url_vectordb_instance = PrepareURLVectorDB(
                urls=urls,
                persist_directory=self.persist_directory,
                embedding_model_engine=self.embedding_model_engine,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap)
            db_stat = prepare_url_vectordb_instance.prepare_and_save_vectordb()
            return db_stat
        else:
            return False


def search_the_requested_url(user_full_query: str) -> List[str]:
    """
    Searches for URLs in the user's query and checks their content types.

    Parameters:
        user_full_query (str): The user's query containing potential URLs.

    Returns:
        List[str]: A list of content types for each URL found in the query.
    """
    url_checker_instance = URLPrep(text=user_full_query)
    status_bool = url_checker_instance.prepare_vector_db()
    return status_bool
