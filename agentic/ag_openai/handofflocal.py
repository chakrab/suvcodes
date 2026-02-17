import os
import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, set_trace_processors
from dotenv import load_dotenv
from localtracer import LocalTracingProcessor

class HandoffLocalAgent:
    def __init__(self, model, ep, key):
        set_trace_processors([LocalTracingProcessor()])

        # Poet Agent
        the_poet = AsyncOpenAI(base_url=ep, api_key=key)
        the_poet_model = OpenAIChatCompletionsModel(model=model, openai_client=the_poet)
        poet_agent = Agent(
            name="Poet Agent",
            handoff_description="If the user requests a poem, hand off to the Poet Agent.",
            instructions="You are a creative poet who writes beautiful poems based on user prompts.",
            model=the_poet_model
        )

        # Story Teller Agent
        the_storyteller = AsyncOpenAI(base_url=ep, api_key=key)
        the_storyteller_model = OpenAIChatCompletionsModel(model=model, openai_client=the_storyteller)
        the_storyteller_agent = Agent(
            name="Short Story Agent",
            handoff_description="If the user requests a short story, hand off to the Short Story Agent.",
            instructions="You are a skilled storyteller who crafts engaging short stories based on user prompts.",
            model=the_storyteller_model
        )

        # Handoff Client
        handoff_client = AsyncOpenAI(base_url=ep, api_key=key)
        handoff_model = OpenAIChatCompletionsModel(model=model, openai_client=handoff_client)
        self.handoff_agent = Agent(
            name="Handoff Agent",
            instructions="You are an agent that hands off user requests to either the Poet Agent or the Short Story Agent based on the content of the request. If the user requests a poem, hand off to the Poet Agent. If the user requests a short story, hand off to the Short Story Agent. If the request is neither, respond appropriately.",
            model=handoff_model,
            handoffs=[poet_agent, the_storyteller_agent]
        )

    async def handle_request(self, text: str) -> str:
        prompt = f"User Request: {text}"
        # Run the handoff agent with the constructed prompt
        response = await Runner.run(self.handoff_agent, prompt)
        print(f"Response: \n{response.final_output}")
        return response


if __name__ == "__main__":
    load_dotenv()

    OLLAMA_MODEL = "llama3.2:3b"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')

    agent = HandoffLocalAgent(OLLAMA_MODEL, OLLAMA_EP, OLLAMA_KEY)
    user_request = "Can you write me a haiku about nature?"
    print(asyncio.run(agent.handle_request(user_request)))