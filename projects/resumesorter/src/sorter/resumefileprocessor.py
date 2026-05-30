import json
import asyncio
import requests
from openai import AsyncOpenAI
from agents import (Agent, Runner, OpenAIChatCompletionsModel, function_tool, 
    RunResult, ToolCallOutputItem, set_tracing_disabled, AgentToolStreamEvent, ItemHelpers)
from pydantic import BaseModel

class CategoryList(BaseModel):
    categories: list[str]

class CategorySelectionInput(BaseModel):
    requirement: str
    categories: list[str]

class RequirementCategory(BaseModel):
    primary_category: str
    alternate_category: str | None

class ResumeProcessor:
    """
    A class to process resumes based on the extracted information from the markdown files.
    The processing is done based on total experience, skills and education.
    """
    resume_service_ep = "http://localhost:8000/api"
    category_ep = f"{resume_service_ep}/get-subjects"
    candidates_ep = f"{resume_service_ep}/get-candidates/{{subject}}"
    resume_ep = f"{resume_service_ep}/get-resume/{{subject}}/{{candidateId}}"

    #OLLAMA_MODEL = "llama3.1:8b"
    OLLAMA_MODEL = "qwen3.5:9b" # Using qwen3.5 for better reasoning capabilities, especially for category selection based on requirements.
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = "dummy"

    def __init__(self):
        #set_trace_processors([LocalTracingProcessor()])
        set_tracing_disabled(True)
        

        # Categorizer Agent
        client = AsyncOpenAI(base_url=ResumeProcessor.OLLAMA_EP, api_key=ResumeProcessor.OLLAMA_KEY)
        model = OpenAIChatCompletionsModel(model=ResumeProcessor.OLLAMA_MODEL, openai_client=client)
        self.categorizer_agent = Agent(
            model=model,
            name="CategorizerAgent",
            instructions="You are a helping agent. Identify one appropriate category for the requirement."
        )

    @staticmethod
    @function_tool
    def get_categories() -> CategoryList:
        """
        A function tool to get the list of available categories from the resume service. The output is a 
        CategoryList object containing the list of available categories.
        """
        # Get all available categories from the resume service
        all_categories = requests.get(ResumeProcessor.category_ep).json().get('subjects', [])
        print(f"Available categories: {all_categories}")
        return CategoryList(categories=all_categories)
    
    @staticmethod
    @function_tool
    def get_resume(category: str, candidate_id: str) -> str:
        """
        A function tool to get the resume for the given candidate ID from the resume service. The input is the 
        candidate ID and the output is the resume in markdown format.
        """
        resume = requests.get(ResumeProcessor.resume_ep.format(subject=category.upper(), candidateId=candidate_id)).json().get('resume', '')
        return resume
    
    @staticmethod
    @function_tool
    def get_candidates_for_category(category: str) -> list[str]:
        """
        A function tool to get the list of candidate IDs for the given category from the resume service. The input is the 
        category and the output is a list of candidate IDs for that category.
        """
        candidates = requests.get(ResumeProcessor.candidates_ep.format(subject=category)).json().get('candidates', [])
        print(f"Total candidates found for category {category}: {len(candidates)}")
        return candidates
    
    @staticmethod
    async def extractor(cat_result: RunResult) -> RequirementCategory:
        """
        A function tool to extract the primary category and alternate category from the result returned by the categorizer agent. 
        The input is the result returned by the categorizer agent and the output is a RequirementCategory object containing the 
        primary category and alternate category.
        """
        print(f"Result from categorizer agent: {cat_result.final_output}")
        for item in reversed(cat_result.new_items):
            if isinstance(item, ToolCallOutputItem) and item.output.strip().startswith("{"):
                try:
                    output_data = json.loads(item.output.strip())
                    primary_category = output_data.get("primary_category", "")
                    alternate_category = output_data.get("alternate_category", "")
                    print(f"Extracted primary category: {primary_category}, alternate category: {alternate_category}")
                    return RequirementCategory(primary_category=primary_category, alternate_category=alternate_category)
                except json.JSONDecodeError:
                    continue
        return RequirementCategory(primary_category="", alternate_category="")
    
    async def handle_stream(self, event: AgentToolStreamEvent) -> None:
        evt_type = event['event'].type
        if evt_type != "raw_response_event":
            if evt_type == "agent_updated_stream_event":
                print(f"[stream] {event['agent'].name} updated")
            elif evt_type == "run_item_stream_event":
                if evt_type == "tool_call_item":
                    print("-- Tool was called")
            elif evt_type == "tool_call_output_item":
                    print(f"-- Tool output: {event.item.output}")
            elif evt_type == "message_output_item":
                    print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                pass  # Ignore other event types
    
    async def process_resume(self, requirement: str):
        instructions = """
            You are an experienced HR professional who is assigned the task of finding candidates for the
            requirement provided below. You have access to the following tools:

            * get_categories: This tool returns the list of available categories from the resume service.
            You can use this tool to get the list of categories. This category can be used to find candidates
                for the requirement.

            * categorizer_agent: This tool should be run after running get_categories. Category selection 
                returned by this tool should always be one of the categories returned by the get_categories tool. 
                This is an agent that can be used to find the most appropriate category for 
                the given requirement. You can use this agent to find the primary category and alternate category 
                for the requirement. The primary category is the most appropriate category for the requirement and 
                the alternate category is the second most appropriate category for the requirement. You can use the 
                get_categories tool to get the list of available categories and then use the categorizer_agent to 
                find the primary and alternate categories for the requirement.

            * get_candidates_for_category: This tool takes a category as input and returns the list of candidate IDs 
                for that category from the resume service. You can use this tool to get the list of candidate IDs for the 
                primary category. For now, you can ignore the alternate category and only focus on the primary category to 
                find candidates for the requirement.

            Use the above tools to find the candidates for the given requirement. Here are the steps you should follow to find 
                the candidates for the requirement:
            
            1. You should first use the get_categories tool to get the list of available categories 
            2. Use the categorizer_agent to find the primary category for the requirement
            3. You should use the get_candidates_for_category tool to get the list of candidate IDs for the primary category 
            4. Return a list of random 5 candidate IDs obtained from the previous step.
        """

        categorizer_instructions = """
        Categorize the requirement into one of the categories obtained from get_categories tool. The output should be a JSON object 
        containing the primary category and alternate category. The primary category is the most appropriate category for the requirement 
        and the alternate category is the second most appropriate category for the requirement. The input to this agent is JSON 
        object containing the requirement and the list of categories obtained from get_categories tool."
        """

        client = AsyncOpenAI(base_url=ResumeProcessor.OLLAMA_EP, api_key=ResumeProcessor.OLLAMA_KEY)
        model = OpenAIChatCompletionsModel(model=ResumeProcessor.OLLAMA_MODEL, openai_client=client)
        agent = Agent(
            name="Resume Processor Agent",
            instructions=instructions,
            model=model,
            tools=[
                ResumeProcessor.get_categories,
                self.categorizer_agent.as_tool(
                    tool_name="categorizer_agent",
                    tool_description=categorizer_instructions,
                    parameters=CategorySelectionInput,
                    include_input_schema=True,
                    custom_output_extractor=ResumeProcessor.extractor,
                    on_stream=self.handle_stream
                ),
                ResumeProcessor.get_candidates_for_category,
                ResumeProcessor.get_resume
            ]
        )
        response = await Runner.run(agent, requirement)
        print(f"Final response from agent: {response.final_output}")
        return True

if __name__ == "__main__":
    resume_processor = ResumeProcessor()
    asyncio.run(resume_processor.process_resume("Find me candidates for a junior developer role withat least 3 years of experience in Python and machine learning."))
