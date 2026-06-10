import logging
from fastmcp import FastMCP
from fastmcp.tools import tool
from fastmcp.resources import resource
from searchfeature import SearchFeature

class SearchMCP():

    def __init__(self) -> None:
        self.port = 8081
        self.name = "Web Search MCP"
        self.mcp = FastMCP(self.name)
        console_handler = logging.StreamHandler()
        logging.basicConfig(
            format="{asctime} - {levelname}: {message}", 
            level=logging.INFO,
            handlers=[console_handler],
            style="{"
        )
        self.logger = logging.getLogger(self.name)

    def start(self):
        self.logger.info(f"Starting MCP on {self.port}")
        self.mcp.add_tool(self.search_web)
        self.mcp.add_resource(self.get_version)
        self.mcp.run(transport="http", host="0.0.0.0", port=self.port)


    @resource("help://about")
    def get_version(self) -> dict[str, str]:
        """
        Just a simple about method
        """
        return {"version": "1.0", "desc": "A sample server", "status": "ok"}

    @tool
    def search_web(self, query:str, max_results:int) -> list:
        """
        Searches web and returns results using DuckDuckGo
        Params:
            :query - Search Text
            :max_results - Max number of results to return
        """
        sf = SearchFeature()
        results = sf.search(query=query, maxresults=max_results) 
        return results

if __name__ == "__main__":
     smcp = SearchMCP()
     smcp.start()
