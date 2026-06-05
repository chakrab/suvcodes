import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, SQLiteSession, set_tracing_disabled
import asyncio


class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get_session(self, session_name: str) -> SQLiteSession:
        """
        Creates a new session for maintaining chat history. The session is stored in a SQLite database with 
        the given session name.
        """
        if session_name not in self.sessions:
            self.sessions[session_name] = SQLiteSession(session_id=session_name)
        return self.sessions[session_name]  
    
    def clear_sessions(self):
        """
        Clears all existing sessions from the session manager.
        """
        self.sessions.clear()

    def delete_session(self, session_name: str):
        """
        Deletes a specific session from the session manager based on the provided session name.
        """
        if session_name in self.sessions:
            del self.sessions[session_name]


class SessionAgent:
    """
    An agent that can maintain chat history across multiple sessions. Each session is stored in a SQLite database,
    allowing the agent to provide context-aware responses based on the conversation history within that session.
    The agent can handle multiple sessions simultaneously, making it suitable for scenarios where multiple users
    are interacting with the agent at the same time.
    """
    def __init__(self, model, ep, key):
        set_tracing_disabled(disabled=True)
        ai_client = AsyncOpenAI(base_url=ep, api_key=key)
        ai_model = OpenAIChatCompletionsModel(model=model, openai_client=ai_client)
        self.chat_agent = Agent(
            name="Session Agent", 
            instructions="You are a helpful assistant that can answer general knowledge questions. Be concise.", 
            model=ai_model
        )

        self.session_manager = SessionManager()

    async def ask_question(self, question: str, session_name: str = ''):
        """
        Asks a question to the agent within a specific session. The session maintains the chat history,
        allowing the agent to provide context-aware responses. The response is returned.
        """
        if (session_name == ''):
            response = await Runner.run(self.chat_agent, question)
        else:
            session = self.session_manager.get_session(session_name)
            response = await Runner.run(self.chat_agent, question, session=session)
        return response

    async def converse_no_session(self):
        """
        Simulates a conversation with the agent, simulating single user asking question without sessions.
        """
        question = ''
        while True:
            question = input("\n\nUSER>>> ")
            if question.lower() != 'bye':
                response = await self.ask_question(question)
                print(f"Response: {response.final_output}")
            else:
                break

    async def converse_session(self):
        """
        Simulates a conversation with the agent, simulating single user asking question in a session.
        """
        question = ''
        while True:
            question = input("\n\nUSER>>> ")
            if question.lower() != 'bye':
                response = await self.ask_question(question, 'default')
                print(f"Response: {response.final_output}")
            else:
                break
        self.session_manager.clear_sessions()

    async def converse_multiple_sessions(self, users: list[str]):
        """
        Simulates a conversation with the agent, simulating multiple users asking question in different sessions.
        Each user has their own session, allowing the agent to maintain separate chat histories for each user
        and provide context-aware responses based on the individual conversations.
        """
        question = ''
        user = ''
        user_csv = ','.join(users)
        while True:
            user = input(f"\n\nUSER ({user_csv})>>> ")
            if user.lower() == 'bye':
                break
            elif user.lower() not in users:
                print(f"Invalid user. Please choose from: {user_csv}")
                continue

            question = input(f"\nQUESTION ({user})>>> ")
            if question.lower() != 'bye':
                response = await self.ask_question(question, session_name=user.lower())
                print(f"Response: {response.final_output}")
            else:
                self.session_manager.delete_session(user)
                print(f"Session for {user} deleted.")
                if all(user not in self.session_manager.sessions for user in users):
                    break
        self.session_manager.clear_sessions()

if __name__ == "__main__":
    load_dotenv()

    OLLAMA_MODEL = "jgmolinawork/gpt5.2-lite"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')
    agent = SessionAgent(OLLAMA_MODEL, OLLAMA_EP, OLLAMA_KEY)

    #print("Starting conversation without session...")
    #asyncio.run(agent.converse_no_session())

    #print("\n\nStarting conversation with session...")
    #asyncio.run(agent.converse_session())

    print("\n\nStarting conversation with multiple sessions...")
    asyncio.run(agent.converse_multiple_sessions(users=['alice', 'bob', 'charlie']))