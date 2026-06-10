from fastmcp import Client, FastMCP
import asyncio

class MCPClient(Client):
    def __init__(self):
        pass

    async def test_resource(self):
        client = Client("http://localhost:8081/mcp")
        async with client:
            print(f"Connected: {client.is_connected()}")
            response = await client.read_resource("help://about")
            print(response)

            response = await client.call_tool("search_web", {"max_results": 2, "query": "Albert Einstein"})
            print(response)

if __name__ == "__main__":
    mcp = MCPClient()
    asyncio.run(mcp.test_resource())
