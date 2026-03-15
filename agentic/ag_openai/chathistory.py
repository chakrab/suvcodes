import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, SQLiteSession, set_tracing_disabled
import asyncio

class ChatHistoryAgent:
    def __init__(self, model, ep, key):
        self.seessions = {}
        self.joe_questions = [
            "What is the capital of France?",
            "What are some historical monuments to see there?",
            "Who are some famous people from there?"
        ]
        self.jane_questions = [
            "How many moons does Saturn have?",
            "Name five of them.",
            "Which one is the largest?"
        ]

        set_tracing_disabled(disabled=True)

        ai_client = AsyncOpenAI(base_url=ep, api_key=key)
        ai_model = OpenAIChatCompletionsModel(model=model, openai_client=ai_client)
        self.chat_agent = Agent(
            name="Session History Agent", 
            instructions="You are a helpful assistant that can answer general knowledge questions. Be concise.", 
            model=ai_model
        )

    def get_session(self, session_name: str) -> SQLiteSession:
        """
        Creates a new session for maintaining chat history. The session is stored in a SQLite database with 
        the given session name.
        """
        if session_name not in self.seessions:
            self.seessions[session_name] = SQLiteSession(session_id=session_name)
        return self.seessions[session_name]
        
    async def ask_question(self, session_name: str, question: str) -> str:
        """
        Asks a question to the agent within a specific session. The session maintains the chat history, 
        allowing the agent to provide context-aware responses. The response is printed and returned.
        """
        if (session_name == ''):
            response = await Runner.run(self.chat_agent, question)
        else:
            session = self.get_session(session_name)
            response = await Runner.run(self.chat_agent, question, session=session)
        print(f"Response: {response.final_output}")
        return response
    
    async def converse_no_session(self):
        """
        Simulates a conversation with the agent, simulating single user asking question without sessions.
        """
        for i in range(len(self.joe_questions)):
            print(f"\n\nJOE>>> {self.joe_questions[i]}")
            await self.ask_question('', self.joe_questions[i])

    async def converse_session(self):
        """
        Simulates a conversation with the agent, simulating single user asking question in a session.
        """
        for i in range(len(self.joe_questions)):
            print(f"\n\nJOE>>> {self.joe_questions[i]}")
            await self.ask_question('joe_session', self.joe_questions[i])
    
    async def converse_multiple(self):
        """
        Simulates a conversation with the agent, simulating multiple users asking question in different sessions.
        """
        for i in range(len(self.joe_questions)):
            print(f"\n\nJOE>>> {self.joe_questions[i]}")
            await self.ask_question("joe_session", self.joe_questions[i])
            print(f"\n\nJANE>>> {self.jane_questions[i]}")
            await self.ask_question("jane_session", self.jane_questions[i])
        
if __name__ == "__main__":
    load_dotenv()

    OLLAMA_MODEL = "jgmolinawork/gpt5.2-lite"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')
    #agent = ChatHistoryAgent(OLLAMA_MODEL, OLLAMA_EP, OLLAMA_KEY)
    agent = ChatHistoryAgent(OLLAMA_MODEL, OLLAMA_EP, OLLAMA_KEY)
    #asyncio.run(agent.converse_no_session())
    asyncio.run(agent.converse_session())
    #asyncio.run(agent.converse_multiple())