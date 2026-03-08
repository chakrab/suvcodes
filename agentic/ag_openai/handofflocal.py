import os
import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, set_trace_processors
from dotenv import load_dotenv
from localtracer import LocalTracingProcessor

class HandoffLocalAgent:
    """
    This class defines a local agent that can hand off user requests to either a Poet Agent or a Short 
    Story Agent based on the content of the request. It uses the OpenAI API for processing the requests 
    and generating responses. The agent is designed to run locally, and it includes tracing capabilities 
    for debugging and analysis purposes.
    """
    def __init__(self, model, ep, key):
        set_trace_processors([LocalTracingProcessor()])

        """
        Poet Agent
         - This agent is responsible for generating poems based on user prompts. It uses the OpenAI API to
         generate responses and is designed to handle requests that are specifically for poems. The agent 
         includes instructions to guide its poetic creation process and is set up to be handed off to by 
         the Handoff Agent when a user request is identified as a request for a poem.
        """
        the_poet = AsyncOpenAI(base_url=ep, api_key=key)
        the_poet_model = OpenAIChatCompletionsModel(model=model, openai_client=the_poet)
        poet_agent = Agent(
            name="Poet Agent",
            handoff_description="If the user requests a poem, hand off to the Poet Agent.",
            instructions="You are a creative poet who writes beautiful poems based on user prompts.",
            model=the_poet_model
        )

        """
        Short Story Agent
         - This agent is responsible for crafting engaging short stories based on user prompts. It uses the
              OpenAI API to generate responses and is designed to handle requests that are specifically for 
              short stories. The agent includes instructions to guide its storytelling process and is set up 
              to be handed off to by the Handoff Agent when a user request is identified as a request for a 
              short story.
        """
        the_storyteller = AsyncOpenAI(base_url=ep, api_key=key)
        the_storyteller_model = OpenAIChatCompletionsModel(model=model, openai_client=the_storyteller)
        the_storyteller_agent = Agent(
            name="Short Story Agent",
            handoff_description="If the user requests a short story, hand off to the Short Story Agent.",
            instructions="You are a skilled storyteller who crafts engaging short stories based on user prompts.",
            model=the_storyteller_model
        )

        """
        Handoff Agent
         - This agent is responsible for receiving user requests and determining which of the two agents 
           (Poet Agent or Short Story Agent) should handle the request based on its content. It uses the 
           same OpenAI model for processing the requests and generating responses, and it includes 
           instructions to guide its decision-making process.
         - The handoff agent is designed to analyze the user's request and route it to the appropriate agent 
           based on whether the request is for a poem or a short story. If the request does not fit either 
           category, it should respond appropriately.
         - The handoff agent also includes tracing capabilities to allow for debugging and analysis of how 
           requests are being processed and routed.
        """
        handoff_client = AsyncOpenAI(base_url=ep, api_key=key)
        handoff_model = OpenAIChatCompletionsModel(model=model, openai_client=handoff_client)
        self.handoff_agent = Agent(
            name="Handoff Agent",
            instructions="You are an agent that hands off user requests to either the Poet Agent or the Short Story Agent based on the content of the request. If the user requests a poem, hand off to the Poet Agent. If the user requests a short story, hand off to the Short Story Agent. If the request is neither, respond appropriately.",
            model=handoff_model,
            handoffs=[poet_agent, the_storyteller_agent]
        )

    async def handle_request(self, text: str) -> str:
        """
        This method takes a user request as input, constructs a prompt for the handoff agent, and runs the 
        agent to get a response. The response is then printed and returned.
        """
        prompt = f"User Request: {text}"
        # Run the handoff agent with the constructed prompt
        response = await Runner.run(self.handoff_agent, prompt)
        print(f"Response: \n{response.final_output}")
        return response


if __name__ == "__main__":
    load_dotenv()

    OLLAMA_MODEL = "llama3.1:8b"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')

    agent = HandoffLocalAgent(OLLAMA_MODEL, OLLAMA_EP, OLLAMA_KEY)
    user_request = "Can you write me a haiku about nature?"
    print(asyncio.run(agent.handle_request(user_request)))