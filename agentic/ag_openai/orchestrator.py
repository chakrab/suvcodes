import os
import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_trace_processors
from dotenv import load_dotenv
from localtracer import LocalTracingProcessor

class Orchestrator:
    """
    The Orchestrator class initializes multiple agents, each specializing in a different domain 
    (Chemistry, History, Astronomy), and a main driver agent that orchestrates the use of these tools to answer
    user questions. The driver agent determines which specialized agent to use based on the user's request and 
    retrieves the appropriate information.
    """
    def __init__(self, model, ep, key):
        set_trace_processors([LocalTracingProcessor()])

        chem_agent = AsyncOpenAI(base_url=ep, api_key=key)
        chem_agent_model = OpenAIChatCompletionsModel(model=model, openai_client=chem_agent)
        chemistry_agent = Agent(
            name="Chemistry Agent",
            instructions="You are an expert in chemistry who provides detailed and accurate information about chemical reactions and properties.",
            model=chem_agent_model
        )

        history_agent = AsyncOpenAI(base_url=ep, api_key=key)
        history_agent_model = OpenAIChatCompletionsModel(model=model, openai_client=history_agent)
        history_agent = Agent(
            name="History Agent",
            instructions="You are a knowledgeable historian who provides detailed and accurate information about historical events, timelines and figures.",
            model=history_agent_model
        )

        astronomy_agent = AsyncOpenAI(base_url=ep, api_key=key)
        astronomy_agent_model = OpenAIChatCompletionsModel(model=model, openai_client=astronomy_agent)
        astronomy_agent = Agent(
            name="Astronomy Agent",
            instructions="You are an expert in astronomy who provides detailed and accurate information about celestial bodies and phenomena.",
            model=astronomy_agent_model
        )

        main_agent = AsyncOpenAI(base_url=ep, api_key=key)
        main_agent_model = OpenAIChatCompletionsModel(model=model, openai_client=main_agent)
        self.driver_agent = Agent(
            name="Driver Agent",
            instructions=
            '''You are a helpful assistant that has access to various tools to answer user questions.
            You can answer questions related to chemistry, history, and astronomy by using the appropriate tools.
            When you receive a user request, determine which tool is best suited to answer the question and
            use that tool to get the answer. If the question is not related to any of the tools, let the user
            know that you cannot answer the question.

            - If a question is related to chemical reactions, properties of elements, or any chemistry-related 
            topic, use the "Chemistry Agent".
            - If a question is related to historical events, figures, or any history-related topic, use the 
            "History Agent".
            - If a question is related to celestial bodies, astronomical phenomena, or any astronomy-related
            topic, use the "Astronomy Agent".
            ''',
            model=main_agent_model,
            tools=[
                chemistry_agent.as_tool(
                    tool_name="Chemistry Agent",
                    tool_description="Use this tool to answer questions related to Chemistry."
                ), history_agent.as_tool(
                    tool_name="History Agent",
                    tool_description="Use this tool to answer questions related to History."
                ), astronomy_agent.as_tool(
                    tool_name="Astronomy Agent",
                    tool_description="Use this tool to answer questions related to Astronomy."
                )
            ]
        )

    async def handle_request(self, text: str) -> str:
        prompt = f"User Request: {text}"
        # Run the handoff agent with the constructed prompt
        response = await Runner.run(self.driver_agent, prompt)
        print(f"Response: \n{response.final_output}")
        return response
    
if __name__ == "__main__":
    load_dotenv()

    OLLAMA_MODEL = "jgmolinawork/gpt5.2-lite"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')

    agent = Orchestrator(OLLAMA_MODEL, OLLAMA_EP, OLLAMA_KEY)
    user_request = "What is the chemistry of cooking noodles?"
    asyncio.run(agent.handle_request(user_request))