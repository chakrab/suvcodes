import os
import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Agent, Runner, OpenAIChatCompletionsModel, trace, set_trace_processors
from localtracer import LocalTracingProcessor

class NDEEventSummary(BaseModel):
    emotions: list[str]
    sensations: list[str]
    recurring_motifs: list[str]
    main_insights: list[str]
    context: str
    others: str

class NDEEventResearch(BaseModel):
    contents: list[NDEEventSummary]

class Research:
    def __init__(self, model, ep):
        load_dotenv()

        self.model = model

        set_trace_processors([LocalTracingProcessor()])
        llm_client = AsyncOpenAI(base_url=ep, api_key=os.getenv("OLLAMA_API_KEY"))
        llm_model = OpenAIChatCompletionsModel(openai_client=llm_client, model=model)

        research_agent_prompt = """
        You are a scientific research assistant. You are provided with list of NDE event summary by participants. 
        Your task is to analyze the contents provided and extract key insights, patterns, and themes
        related to the NDE experiences. You should focus on identifying common elements across the narratives, 
        such as emotions, sensations, and any recurring motifs. Additionally, you should consider the context 
        of the events and how they relate to the participants' backgrounds and beliefs. Your analysis should be
        thorough and detailed, providing a comprehensive understanding of the NDE experiences as described in the
        uploaded file. Additionally, you should provide a summary of the key findings and insights derived from 
        the analysis. Document your findings in a clear and concise manner, ensuring that your analysis is 
        well-structured and easy to understand.
        """

        summary_agent_prompt = """
        You are a summarizing assistant. You are provided with NDE events as narrated by participants. 
        Your task is to analyze the content of the uploaded file and extract key insights, patterns, and themes
        related to the NDE experiences. Write at least 5 bullet points summarizing the key findings from the 
        narratives. Focus on identifying the key elements for the narrative and document in third person perspective.
        """

        self.summary_agent = Agent(
            name="Summary Agent",
            instructions=summary_agent_prompt,
            model=llm_model,
            output_type=NDEEventSummary
        )
        self.research_agent = Agent(
            name="Research Agent",
            instructions=research_agent_prompt,
            model=llm_model
        )
    
    def get_files_content(self, file_paths):
        contents = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                contents.append(f.read())
        return contents 

    async def summarize(self, content):
        print("Running Summarization...")
        with trace("Summary Agent Trace"):
            try:
                response = await Runner.run(self.summary_agent, content)
                print("Summary completed.")
                return response.final_output
            except Exception as e:
                print(f"Error running summary agent: {e}")
                raise

    async def analyze(self, research_input):
        print("Running research...")
        with trace("Research Agent Trace"):
            try:
                response = await Runner.run(self.research_agent, research_input)
                print("Research completed.")
                return response
            except Exception as e:
                print(f"Error running research agent: {e}")
                raise

    async def main(self, file_contents):
        tasks = []
        for content in file_contents:
            tasks.append(asyncio.create_task(self.summarize(content)))
        results = await asyncio.gather(*tasks)
        print("Summarization tasks completed.")
        """
        print(f"\nGenerated {len(results)} Summary:\n")
        for i, result in enumerate(results):
            print(f"Summary {i+1}:\n")
            print(result)
            print("\n" + "="*80 + "\n")
        """
        dict_in = NDEEventResearch(contents=results).model_dump_json()
        analysis_out = await self.analyze(dict_in)
        print("\nGenerated Research Analysis:\n")
        print(analysis_out)

if __name__ == "__main__":
    yt_files = [
        "../data/Female_001.txt",
        "../data/Female_002.txt",
        "../data/Female_003.txt",
        "../data/Male_001.txt",
        "../data/Male_002.txt",
        "../data/Male_003.txt"
    ]
    base_model = "qwen3-vl:8b-instruct"
    endpoint = "http://localhost:11434/v1"
    research = Research(model=base_model, ep=endpoint)
    file_contents = research.get_files_content(yt_files)
    asyncio.run(research.main(file_contents))
