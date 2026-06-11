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

            all_tools = await client.list_tools()
            print(all_tools)

            all_prompts = await client.list_prompts()
            print(all_prompts)

            all_resources = await client.list_resources()
            print(all_resources)

            response = await client.read_resource("help://about")
            print(response)

            response = await client.call_tool("search_web", {"limit": 1, "q": "Albert Einstein"})
            print(response)

if __name__ == "__main__":
    mcp = MCPClient(transport="http")
    asyncio.run(mcp.test_mcp_service())
