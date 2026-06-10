from ddgs import DDGS

class SearchFeature:
    def __init__(self):
        pass

    def search(self, query, maxresults):
        with DDGS() as ddgs:
            result = ddgs.text(query=query, max_results=maxresults)
        return result
    
if __name__ == "__main__":
    sf = SearchFeature()
    result = sf.search("Charlies Angels", 3)
    for r in result:
        print(r) # [title, href, body]
