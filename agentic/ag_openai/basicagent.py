import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
import asyncio

class BasicAgent:
    """
    A basic agent for creating haikus/tankas.
    """
    def __init__(self, model, ep, key):
        self.style = "Haiku"
        #self.style = "Tanka"
        set_tracing_disabled(disabled=True)

        ai_client = AsyncOpenAI(base_url=ep, api_key=key)
        ai_model = OpenAIChatCompletionsModel(model=model, openai_client=ai_client)
        self.agent = Agent(name=f"{self.style} Agent", instructions=f"You are a Great Master of {self.style} who can create {self.style}s on any subject.", model=ai_model)

    async def create_poetry(self, subject: str) -> str:
        prompt = f"Create a {self.style} about the following subject:\n\n{subject}. Also, explain the meaning of this {self.style}."
        response = await Runner.run(self.agent, prompt)
        print(f"{self.style}:\n{response.final_output}")
        return response

if __name__ == "__main__":
    load_dotenv()

    OLLAMA_MODEL = "command-r7b:7b"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')

    agent = BasicAgent(OLLAMA_MODEL, OLLAMA_EP, OLLAMA_KEY)
    subject = "a candle in the dark"
    asyncio.run(agent.create_poetry(subject))
