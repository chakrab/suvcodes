import logging
from fastmcp import FastMCP
from fastmcp.tools import tool
from fastmcp.resources import resource
from fastmcp.prompts import prompt
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.server.middleware.logging import LoggingMiddleware

from searchfeature import SearchFeature

class CustomLoggingMiddleware(LoggingMiddleware):
    """
    Custom logging through Python loggers, see what request is coming in
    """
    def __init__(self) -> None:
        super().__init__()

        console_handler = logging.StreamHandler()
        logging.basicConfig(
            format="{asctime} - {levelname}: {message}", 
            level=logging.INFO,
            handlers=[console_handler],
            style="{"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    async def on_message(self, context: MiddlewareContext, call_next):
        """
        call_next lets the chain continue after intercept
        """
        self.logger.info(f"→ {context.method} from {context.source}")
        self.logger.info(f"{context.message}")
        result = await call_next(context)
        self.logger.info(f"← {context.method} to {context.source}")
        return result

class SearchMCP():
    """
    This is a basic MCP service that can be used for searching web.
    It will run on port 8081 (hardcoded).

    It does not use annotations (decorators) to define the tools,
        but uses registering direct through API.
    """
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
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def start(self):
        """
        Add the resources/ tools and start the MCP server
        """
        self.logger.info(f"Starting MCP on {self.port}")
        self.mcp.add_middleware(CustomLoggingMiddleware())
        self.mcp.add_tool(self.search_web)
        self.mcp.add_resource(self.get_version)
        self.mcp.add_prompt(self.research)
        self.mcp.run(transport="streamable-http", host="0.0.0.0", port=self.port)

    @resource("help://about")
    def get_version(self) -> dict[str, str]:
        """
        Just a simple about method
        """
        return {"version": "1.0", "desc": "A sample server", "status": "ok"}

    @tool
    def search_web(self, q:str, limit:int) -> list:
        """
        Searches web and returns results using DuckDuckGo
        Params:
            q:      Search Text
            limit:  Max number of results to return
        """
        sf = SearchFeature()
        results = sf.search(query=q, limit=limit) 
        return results
    
    @prompt
    def research(self, topic:str, focus:str = "") -> str:
        """
        Provides a prompt to LLM that can help research
        """
        if focus != "":
            text = f"Do a thorough research on {topic} and summarize response. Do not infer anything not in result. Put more stress on {focus}"
        else:
            text = f"Do a thorough research on {topic} and summarize response. Do not infer anything not in result"
        
        return text

if __name__ == "__main__":
     smcp = SearchMCP()
     smcp.start()
