import logging
from ddgs import DDGS

class SearchFeature:
    def __init__(self):
        console_handler = logging.StreamHandler()
        logging.basicConfig(
            format="{asctime} - {levelname}: {message}", 
            level=logging.INFO,
            handlers=[console_handler],
            style="{"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    """
    This is used for calling DDGS for searching web and getting search results
    """
    def search(self, query, limit):
        """
        Main function to search.
        Param:
            query:      Query
            maxresults: Max results to return
        """
        with DDGS() as ddgs:
            result = ddgs.text(query=query, max_results=limit)
        return result
