import json
import asyncio
import random
import requests
from openai import AsyncOpenAI
from agents import (Agent, Runner, OpenAIChatCompletionsModel, function_tool, 
    RunResult, ToolCallOutputItem, set_tracing_disabled, AgentToolStreamEvent)
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

class CategoryList(BaseModel):
    categories: list[str]

class CategorySelectionInput(BaseModel):
    requirement: str
    categories: list[str]

class RequirementCategory(BaseModel):
    primary_category: str
    alternate_category: str | None

class KeywordScoreInput(BaseModel):
    requirement_keywords: list[str]
    resume_markdown: str

class ResumeProcessor:
    """
    A class to process resumes based on the extracted information from the markdown files.
    The processing is done based on total experience, skills and education.
    """
    resume_service_ep = "http://localhost:8000/api"
    category_ep = f"{resume_service_ep}/get-subjects"
    candidates_ep = f"{resume_service_ep}/get-candidates/{{subject}}"
    resume_ep = f"{resume_service_ep}/get-resume/{{candidateId}}"

    OLLAMA_MODEL = "qwen3-vl:8b-instruct"
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
        self.keyword_agent = Agent(
            model=model,
            name="KeywordAgent",
            instructions="You are a helping agent. Identify the relevant keywords from the requirement that can be used to match with the resumes. Only return keywords, nothing else. The output should be a list of keywords."
        )
        self.keyword_scoring_agent = Agent(
            model=model,
            name="KeywordScoringAgent",
            instructions="You are a helping agent. Given a set of keywords and resume, score the relevance of resume between 1 and 100 where 100 is a perfect match. The output should be a JSON object containing the score and candidate ID. no other information should be returned."
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
        print(f"\nAvailable categories: {all_categories}")
        return CategoryList(categories=all_categories)

    @staticmethod
    @function_tool
    def get_resume(candidate_id: str) -> str:
        """
        A function tool to get the resume for the given candidate ID from the resume service. The input is the 
        candidate ID and the output is the resume in markdown format.
        """
        resume = requests.get(ResumeProcessor.resume_ep.format(candidateId=candidate_id)).json().get('resume', '')
        return resume

    @staticmethod
    @function_tool
    def get_candidates_for_category(category: str) -> list[str]:
        """
        A function tool to get the list of candidate IDs for the given category from the resume service. The input is the 
        category and the output is a list of candidate IDs for that category.
        """
        candidates = requests.get(ResumeProcessor.candidates_ep.format(subject=category)).json().get('candidates', [])
        print(f"\nTotal candidates found for category {category}: {len(candidates)}")
        return candidates
    
    @staticmethod
    @function_tool
    def get_n_random_candidates_for_category(category: str, n: int) -> list[str]:
        """
        A function tool to get a list of n random candidate IDs for the given category from the resume service. The input is the 
        category and the number of random candidates to return, and the output is a list of n random candidate IDs for that category.
        """
        candidates = requests.get(ResumeProcessor.candidates_ep.format(subject=category)).json().get('candidates', [])
        print(f"\nTotal candidates found for category {category}: {len(candidates)}")
        if len(candidates) <= n:
            return candidates
        else:
            return random.sample(candidates, n)

    @staticmethod
    async def extractor(cat_result: RunResult) -> RequirementCategory:
        """
        A function tool to extract the primary category and alternate category from the result returned by the categorizer agent. 
        The input is the result returned by the categorizer agent and the output is a RequirementCategory object containing the 
        primary category and alternate category.
        """
        print(f"\nResult from categorizer agent: {cat_result.final_output}")
        return RequirementCategory(primary_category=cat_result.final_output, alternate_category="")
    
    @staticmethod
    async def score_extractor(score_result: RunResult) -> RunResult:
        """
        A function tool to extract the relevance score and candidate ID from the result returned by the keyword scoring agent. 
        The input is the result returned by the keyword scoring agent and the output is a JSON object containing the candidate ID
        and relevance score.
        """
        print(f"\nResult from keyword scoring agent: {score_result.final_output}")
        return score_result.final_output

    async def handle_stream(self, event: AgentToolStreamEvent) -> None:
        evt_type = event['event'].type
        if evt_type != "raw_response_event":
            if evt_type == "agent_updated_stream_event":
                print(f"\n[stream] {event['agent'].name} updated")
            elif evt_type == "run_item_stream_event":
                if evt_type == "tool_call_item":
                    print("\n-- Tool was called")
                elif evt_type == "tool_call_output_item":
                    print(f"\n-- Tool output: {event.item.output}")
                elif evt_type == "message_output_item":
                    pass # We are already printing the delta output
        else:
            if isinstance(event['event'].data, ResponseTextDeltaEvent):
                print(f"{event['event'].data.delta}", end="", flush=True)

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

            * get_n_random_candidates_for_category: This tool takes a category and number n as input and returns a list of n 
                random candidate IDs for that category from the resume service. You can use this tool to get a list of 5 
                random candidate IDs for the primary category.

            * keyword_agent: This is an agent that can be used to extract relevant keywords from the requirement. You can use 
                this agent to extract the relevant keywords from the requirement that can be used to match with the resumes. 
                The output of this agent is a list of relevant keywords extracted from the requirement.

            * get_resume: This tool takes a candidate ID as input and returns the resume for that candidate ID from 
                the resume service.

            * keyword_scoring_agent: This is an agent that can be used to score the relevance of a resume for the requirement based 
                on the presence of relevant keywords in the resume. It takes a list of relevant keywords and a resume in markdown format 
                as input and returns a relevance score between 1 and 100 where 100 is a perfect match. You can use this agent to score the 
                relevance of the resumes for the requirement based on the presence of relevant keywords in the resumes.

            Use the above tools to find the candidates for the given requirement. Here are the steps you should follow to find 
                the candidates for the requirement:
            
            1. You should first use the get_categories tool to get the list of available categories 
            2. Use the categorizer_agent to find the primary category for the requirement
            3. Use the keyword_agent to extract the relevant keywords from the requirement that can be used to match with the resumes. 
                This will help you to find the candidates that are more relevant to the requirement.
            4. You should use the get_n_random_candidates_for_category tool to get the list of 5 random candidate IDs for the primary category 
            5. For each candidate ID from the list, use the get_resume tool to get the resume for that candidate ID and check if the 
                resume contains the relevant keywords extracted from the requirement. You can use the keyword_scoring_agent to score 
                the relevance of the resume for the requirement based on the presence of relevant keywords in the resume.
            7. Finally, return the list of candidate IDs with scores that are a good match for the requirement based on the presence
                of relevant keywords in their resumes.
        """

        categorizer_instructions = """
        Categorize the requirement into one of the categories obtained from get_categories tool. The output should be a JSON object 
        containing the primary category and alternate category. The primary category is the most appropriate category for the requirement 
        and the alternate category is the second most appropriate category for the requirement. The input to this agent is JSON 
        object containing the requirement and the list of categories obtained from get_categories tool."
        """

        keyword_instructions = """
        Identify the relevant keywords from the requirement that can be used to match with the resumes. The output should
        be a list object containing the list of relevant keywords. The input to this agent is the requirement string.
        """

        scoring_instructions = """
        Given a set of keywords and resume, score the relevance of resume between 1 and 100 where 100 is a perfect match. The output 
        should be a score. The input to this agent is a JSON object containing the list of relevant keywords and the resume in markdown 
        format. Use the presence of relevant keywords in the resume to determine the relevance of the resume for the requirement. If the 
        resume contains most of the relevant keywords, then the score should be high. If the resume contains few or none of the relevant
        keywords, then the score should be low.

        Resume is in JSON format with the following structure:
        {"resume": [{"total_experience":"5 years","skills":["Python","Machine Learning"],"education":"Bachelor's in Computer Science"}, ...], "subject":"INFORMATION-TECHNOLOGY","candidate":"26768723"}]}

        Use the total experience, skills and education information from the resume to determine the relevance of the resume for the requirement. 
        For example, if the requirement is for a junior developer role with at least 3 years of experience   in Python and machine learning, then 
        a resume with 5 years of experience in Python and machine learning should get a high score, while a resume with 1 year of experience in
        Python and machine learning should get a low score.

        Output should be a JSON object containing the candidate ID and the relevance score. No other information should be included in the output.
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
                self.keyword_agent.as_tool(
                    tool_name="keyword_agent",
                    tool_description=keyword_instructions,
                    include_input_schema=False,
                    custom_output_extractor=None,
                    on_stream=self.handle_stream
                ),
                self.keyword_scoring_agent.as_tool(
                    tool_name="keyword_scoring_agent",
                    tool_description=scoring_instructions,
                    include_input_schema=False,
                    custom_output_extractor=None,
                    on_stream=self.handle_stream
                ),
                ResumeProcessor.get_candidates_for_category,
                ResumeProcessor.get_n_random_candidates_for_category,
                ResumeProcessor.get_resume
            ]
        )
        response = await Runner.run(agent, requirement)
        print(f"\nFinal response from agent: {response}")

        usage = response.context_wrapper.usage
        print(f"\nRequests: {usage.requests}, Total tokens used: {usage.total_tokens}, Prompt tokens: {usage.input_tokens}, Completion tokens: {usage.output_tokens}")
        return True

if __name__ == "__main__":
    resume_processor = ResumeProcessor()
    asyncio.run(resume_processor.process_resume("I have a automotive dealership. I am looking for an experienced digital marketer who can help me with my online marketing campaigns. The ideal candidate should have at least 5 years of experience in digital marketing, with a focus on social media marketing and search engine optimization. They should also have experience working in the automotive industry and be familiar with the latest trends and best practices in digital marketing for automotive dealerships."))
