from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, set_trace_processors, function_tool
from dotenv import load_dotenv
import os
import asyncio

class ToolUsage:
    """
    The ToolUsage class demonstrates how to use a tool within an agent to fetch Chinese Zodiac personality 
    traits based on a birth year. It initializes an agent with the necessary model and tools, and provides a 
    method to run the agent with a given birth year.

    This calculation is based on the traditional Chinese Zodiac system, which assigns an animal and an element 
    to each year in a 12-year cycle. The personality traits are derived from the assigned animal and element.
    However, calculating with only the birth year may not be entirely accurate for determining the Chinese 
    Zodiac sign, as the Chinese do not use a solar calendar, but instead uses a lunar calendar. 
    """
    def __init__(self, model, ep, key):
        set_tracing_disabled(True)
        agent = AsyncOpenAI(base_url=ep, api_key=key)
        agent_model = OpenAIChatCompletionsModel(model=model, openai_client=agent)

        self.zodiac_agent = Agent(
            name="Zodiac Agent",
            instructions="You are a helpful assistant who uses get_chinese_personality_traits tool to provide "\
                "personality traits. Be concise and informative in your responses.",
            model=agent_model,
            tools=[ToolUsage.get_chinese_personality_traits]
        )

    @staticmethod
    @function_tool
    def get_chinese_personality_traits(birth_year: int):
        """
        Get the Chinese Zodiac personality traits based on the birth year.
        """
        print("Fetching Chinese Zodiac personality traits for year:", birth_year)
        elements = ['Wood', 'Fire', 'Earth', 'Metal', 'Water']
        element = elements[(birth_year - 4) % 10 // 2]

        traits = [
            "猴: Monkey - Sharp, smart, curiosity",
            "鸡: Rooster - Observant, hardworking, courageous",
            "狗: Dog - Lovely, honest, prudent",
            "猪: Pig - Compassionate, generous, diligent",
            "鼠: Rat - Quick-witted, resourceful, versatile, kind",
            "牛: Ox - Diligent, dependable, strong, determined",
            "虎: Tiger - Brave, confident, competitive",
            "兔: Rabbit - Quiet, elegant, kind, responsible",
            "龙: Dragon - Confident, intelligent, enthusiastic",
            "蛇: Snake - Enigmatic, intelligent, wise",
            "马: Horse - Animated, active, energetic",
            "羊: Goat - Calm, gentle, sympathetic"
        ]
        trait = traits[birth_year % 12]
        response = {
            "year": birth_year,
            "traits": trait,
            "element": element,
            "description": f"People born in the year {birth_year} are associated with the {element} element and have the following personality traits: {trait}."
        }
        print("Generated response:", response)
        return response
    
    async def run_agent(self, birth_year: int):
        """
        Run the agent to get Chinese Zodiac personality traits for the given birth year.
        """
        prompt = f"What are the Chinese zodiac personality traits for someone born in the year {birth_year}?"
        response = await Runner.run(self.zodiac_agent, prompt)
        print(f"Agent Response: {response.final_output}")
        return response
    
if __name__ == "__main__":
    load_dotenv()

    OLLAMA_MODEL = "llama3.1:8b"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')

    ta = ToolUsage(model=OLLAMA_MODEL, ep=OLLAMA_EP, key=OLLAMA_KEY)
    year = 2004
    asyncio.run(ta.run_agent(year))
