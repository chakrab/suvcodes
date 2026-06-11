from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIResponsesModel, set_trace_processors
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from dotenv import load_dotenv
import os
import asyncio
from localtracer import LocalTracingProcessor

class MCPCaller:
    def __init__(self, model, ep, key) -> None:
        set_trace_processors([LocalTracingProcessor()])
        agent = AsyncOpenAI(base_url=ep, api_key=key)
        self.agent_model = OpenAIResponsesModel(model=model, openai_client=agent)

    async def init_agent(self):
        """
        Run the agent to get information about US/ Iran war. This is current affairs, so should be
        unavialable to the agent.
        """
        server = MCPServerStreamableHttp(
            name="Web Search MCP",
            params={
                "url": "http://localhost:8081/mcp",
                "timeout": 10,
            },
            max_retry_attempts=3,
        )

        helper_agent = Agent(
            name="Helper Agent",
            instructions="You are a helpful assistant for general queries. When needed, you will use the MCP server to get data from web.",
            model=self.agent_model,
            mcp_servers=[server],
            model_settings=ModelSettings(tool_choice="required")
        )

        try:
            _ = await server.connect()
        except asyncio.CancelledError:
            print("Connect Cancelled")
    
        prompt = "What is the current state of war between US and Iran?"
        try:
            response = await Runner.run(helper_agent, prompt)
            print(f"Agent Response: {response.final_output}")
        except asyncio.CancelledError:
            print("Agent task Cancelled")
        return response
    
    async def run_agent(self):
        async with MCPServerStreamableHttp(
            name="Web Search MCP",
            params={
                "url": "http://localhost:8081/mcp",
                "timeout": 10,
            },
            max_retry_attempts=3,
        ) as server:
            await server.connect()
            helper_agent = Agent(
                name="Helper Agent",
                instructions="You are a helpful assistant for general queries. When needed, you will use the MCP server to get data from web.",
                model=self.agent_model,
                mcp_servers=[server],
                model_settings=ModelSettings(tool_choice="required")
            )
            prompt = "What is the current state of war between US and Iran?"
            response = await Runner.run(helper_agent, prompt)
            print(f"Agent Response: {response.final_output}")

if __name__ == "__main__":
    load_dotenv()

    OLLAMA_MODEL = "llama3.1:8b"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')

    ta = MCPCaller(model=OLLAMA_MODEL, ep=OLLAMA_EP, key=OLLAMA_KEY)
    asyncio.run(ta.run_agent())