from fastmcp import Client
import asyncio

class MCPClient(Client):
    async def test_mcp_service(self):
        """
        Tester for the MCP service. Creates a client and runs the methods.
        """
        client = Client("http://localhost:8081/mcp")
        async with client:
            print(f"Connected: {client.is_connected()}")
            response = await client.read_resource("help://about")
            print(response)

            response = await client.call_tool("search_web", {"max_results": 2, "query": "Albert Einstein"})
            print(response)

if __name__ == "__main__":
    mcp = MCPClient()
    asyncio.run(mcp.test_mcp_service())
