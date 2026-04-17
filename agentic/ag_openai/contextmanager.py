"""
We will build a Travel Recommendation System that provides personalized travel suggestions based on user 
preferences and current location. The system will utilize a Context Manager to manage user preferences and 
current context, allowing it to generate tailored travel recommendations.
"""
import os
import asyncio
from openai import AsyncOpenAI
from agents import Agent, RunContextWrapper, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from dotenv import load_dotenv
from dataclasses import dataclass

@dataclass
class UserPreference:
    """
    A dataclass to represent user preferences.
    """
    user_id: int
    name: str
    favorite_sports: str
    favorite_food: str

@dataclass
class UserContext:
    """
    A dataclass to represent the user's context.
    """
    user_id: int

"""
A dummy user preference manager is created to simulate fetching user preferences. In a real-world application, 
this would likely involve database queries or API calls to retrieve user data. The ContextManager class is
responsible for managing user preferences and generating travel recommendations based on those preferences. 
It uses an agent to fetch user preferences and generate recommendations accordingly.
"""
class DummyUserPreferenceManager:
    """
    A dummy manager to simulate fetching user preferences.
    """
    def __init__(self):
        self.user_profs = [
            UserPreference(user_id=1, name="Alice", favorite_sports="Tennis", favorite_food="Pizza"),
            UserPreference(user_id=2, name="Bob", favorite_sports="Football", favorite_food="Sushi"),
            UserPreference(user_id=3, name="Charlie", favorite_sports="Basketball", favorite_food="Burgers"),
            UserPreference(user_id=4, name="Diana", favorite_sports="Swimming", favorite_food="Salad"),
            UserPreference(user_id=5, name="Eve", favorite_sports="Rugby", favorite_food="Pasta")
        ]

    def get_user_preference(self, user_id: int) -> str:
        up = next((prof for prof in self.user_profs if prof.user_id == user_id), None)
        if up:
            print(f"Fetched user preferences for user_id {user_id}: {up}")
            return f"User {up.name} prefers {up.favorite_sports} in sports, and in food prefers {up.favorite_food}."
        else:
            return "User not found."

"""
The ContextManager class is responsible for managing user preferences and generating travel 
recommendations based on those preferences.
"""
class ContextManager:
    """
    The ContextManager class manages user preferences and generates travel recommendations based on those preferences.
    """
    def __init__(self, model, ep, key):
        set_tracing_disabled(True)
        agent = AsyncOpenAI(base_url=ep, api_key=key)
        agent_model = OpenAIChatCompletionsModel(model=model, openai_client=agent)

        self.user_preference_agent = Agent[UserContext](
            name="Travel Manager Agent",
            instructions="You are a helpful assistant who provides tailored travel preferences for the user.",
            model=agent_model,
            tools=[ContextManager.get_user_preferences]
        )

    @function_tool
    @staticmethod
    def get_user_preferences(wrapper:RunContextWrapper[UserContext]) -> str:
        """
        This function is used for fetching user preferences.
        It will be called by the agent when it needs to access the user's preferences based on their user_id.
        """
        dummy_manager = DummyUserPreferenceManager()
        return dummy_manager.get_user_preference(wrapper.context.user_id)
    
    async def run(self, user_id: int, location: str):
        """
        Run the Context Manager to generate a travel recommendation based on user preferences.
        """
        user_info = UserContext(user_id=user_id)
        result = await Runner.run(  
            starting_agent=self.user_preference_agent,
            input=f"Based on the user's food and sports preferences, can you provide a travel recommendation in {location}?",
            context=user_info        
        )
        print(result.final_output)

# Testing the Context Manager
if __name__ == "__main__":
    load_dotenv()

    OLLAMA_MODEL = "llama3.1:8b"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')

    ta = ContextManager(model=OLLAMA_MODEL, ep=OLLAMA_EP, key=OLLAMA_KEY)
    asyncio.run(ta.run(3, "Portugal"))
