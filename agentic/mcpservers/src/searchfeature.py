from ddgs import DDGS

class SearchFeature:
    """
    This is used for calling DDGS for searching web and getting search results
    """
    def search(self, query, maxresults):
        """
        Main function to search.
        Param:
            query:      Query
            maxresults: Max results to return
        """
        with DDGS() as ddgs:
            result = ddgs.text(query=query, max_results=maxresults)
        return result
    
if __name__ == "__main__":
    sf = SearchFeature()
    result = sf.search("Charlies Angels", 3)
    for r in result:
        print(r) # [title, href, body]
