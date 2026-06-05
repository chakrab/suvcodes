import asyncio
from openai import AsyncOpenAI
from agents import (Agent, Runner, OpenAIChatCompletionsModel, ModelSettings, function_tool, set_trace_processors)
from localtracer import LocalTracingProcessor

class TestTracer:
    OLLAMA_MODEL = "jgmolinawork/gpt5.2-lite:latest"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = "dummy"

    def __init__(self):
        set_trace_processors([LocalTracingProcessor()])

        keyword_desc = """
        Extract keywords from the given topic. Return as JSON with 'topic' and 'keywords' fields.
        """
        keyword_client= AsyncOpenAI(base_url=self.OLLAMA_EP, api_key=self.OLLAMA_KEY)
        keyword_model = OpenAIChatCompletionsModel(openai_client=keyword_client, model=self.OLLAMA_MODEL)
        self.keyword_agent = Agent(
            name="KeywordAgent",
            model=keyword_model,
            instructions=keyword_desc
        )

        podcast_desc = """
        Given a topic and keywords, write a podcast suggestion with 'podcast_name', 'episode_title' and 'podcast_text' fields. Return as JSON.
        Just return the JSON, no explanations.
        """
        podcast_client = AsyncOpenAI(base_url=self.OLLAMA_EP, api_key=self.OLLAMA_KEY)
        podcast_model = OpenAIChatCompletionsModel(openai_client=podcast_client, model=self.OLLAMA_MODEL)
        self.podcast_agent = Agent(
            name="PodcastAgent",
            model=podcast_model,
            instructions=podcast_desc
        )

        main_desc = """
        You are an orchestrating service that creates a podcast based on a given topic. You will always use the available tools to accomplish 
        this task. You have access to the following tools:
        
        1) KeywordAgent: to extract keywords from the content. It takes a text as input and returns keywords in JSON format.
        2) PodcastAgent: to generate a podcast suggestion based on the topic and keywords. It returns a JSON with 'podcast_name', 'episode_title' and 'podcast_text' fields.
        3) write_audio: to simulate writing the podcast text to an audio file. It takes 'podcast_name', 'episode_title' and 'podcast_text' as input and creates a text file named [podcast_name]_[episode_title].txt with the podcast text. It returns the filename in JSON format.
        
        Your task is to read the content, extract keywords using KeywordAgent, generate a podcast suggestion using PodcastAgent, and simulate 
        writing the podcast text to an audio file using write_audio. Just return the final response from the main agent, no explanations.
        """
        main_client = AsyncOpenAI(base_url=self.OLLAMA_EP, api_key=self.OLLAMA_KEY)
        main_model = OpenAIChatCompletionsModel(openai_client=main_client, model=self.OLLAMA_MODEL)
        self.main_agent = Agent(
            name="MainAgent",
            model=main_model,
            instructions=main_desc,
            tools=[
                self.keyword_agent.as_tool(
                    tool_name="KeywordAgent",
                    tool_description="Use this tool to extract keywords from a topic. Return as JSON with 'topic' and 'keywords' fields."
                ),
                self.podcast_agent.as_tool(
                    tool_name="PodcastAgent",
                    tool_description="Use this tool to generate podcast suggestions. Return as JSON with 'podcast_name', 'episode_title' and 'podcast_text' fields."
                ),
                TestTracer.write_audio
            ],
            model_settings=ModelSettings(
                tool_choice="required"
            )
        )

    @staticmethod
    @function_tool
    async def write_audio(podcast_name, episode_title, podcast_text):
        """
        Write the podcast text to a file named [podcast_name]_[episode_title].txt to simulate audio conversion. In a real implementation, 
        this would generate an audio file.
        """
        filename = f"{podcast_name}_{episode_title}.txt"
        with open(filename, "w") as f:
            f.write(podcast_text)
        print(f"Simulated audio file created: {filename}")
        return {"filename": filename}

    async def handle_request(self, text: str):
        # Read the file content to use as input for the main agent
        with open(text, "r") as f:
            file_content = f.read()
        prompt = f"\nCreate a podcast based on the following content: {file_content}"
        response = await Runner.run(self.main_agent, prompt)
        print(f"Response: \n{response}")
        return response
    
if __name__ == "__main__":
    test_tracer = TestTracer()
    asyncio.run(test_tracer.handle_request("/Users/suvendra/Downloads/Nowcasting_Econ-Report-v11.md"))
