import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
import asyncio

class FirstAgent:
    """
    This class defines a simple agent that translates English text to Spanish using an OpenAI model. The 
    agent is initialized with the model name, endpoint, and API key. The translate method constructs a 
    prompt and runs the agent to get the Spanish translation of the input English text.
    """
    def __init__(self, model, ep, key):
        set_tracing_disabled(disabled=True)

        ai_client = AsyncOpenAI(base_url=ep, api_key=key)
        ai_model = OpenAIChatCompletionsModel(model=model, openai_client=ai_client)
        self.spanish_agent = Agent(name="Spanish Translator Agent", instructions="You are a translator who can translate from English to Spanish. Be concise.", model=ai_model)

    async def translate(self, text: str) -> str:
        """
        Agent to translate English text to Spanish. It constructs a prompt with the input text and runs the 
        agent to get the translation. The final output is printed and returned.
        """
        prompt = f"Translate the following English text to Spanish:\n\n{text}"
        # Run the agent with the constructed prompt
        response = await Runner.run(self.spanish_agent, prompt)
        print(f"Spanish Translation: {response.final_output}")
        return response
    
if __name__ == "__main__":
    load_dotenv()

    GROQ_MODEL = "groq/compound-mini"
    GROQ_EP = "https://api.groq.com/openai/v1"
    GROQ_KEY = os.getenv('GROQ_API_KEY')

    OLLAMA_MODEL = "command-r7b:7b"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')
    #agent = FirstAgent(OLLAMA_MODEL, OLLAMA_EP, OLLAMA_KEY)
    agent = FirstAgent(GROQ_MODEL, GROQ_EP, GROQ_KEY)
    english_text = "It is a beautiful day to learn about AI agents!"
    asyncio.run(agent.translate(english_text))

"""
GROQ :::
Spanish Translation: ¡Es un día hermoso para aprender sobre agentes de IA!
OPENAI_API_KEY is not set, skipping trace export

OLLAMA :::
Spanish Translation: ¡Qué hermoso día para aprender sobre agentes de intelligence artificial!
OPENAI_API_KEY is not set, skipping trace export
"""