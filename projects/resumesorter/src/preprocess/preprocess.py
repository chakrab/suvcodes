import os
import re
import sys
import json
import asyncio
from pathlib import Path
from tinydb import TinyDB, Query

current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent.parent.parent
sys.path.append(str(parent_directory))
from tracer.localtracer import LocalTracingProcessor

from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_trace_processors, set_tracing_disabled
from dotenv import load_dotenv

class PreprocessResume:
    """
    A class to preprocess resumes in markdown format and extract relevant information using an agent.
    The extracted information is stored in a TinyDB database.
    """
    def __init__(self, src_dir: Path):
        OLLAMA_MODEL = "mistral:7b"
        OLLAMA_EP = "http://localhost:11434/v1"
        OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')

        self.tinydb = TinyDB('../../data/db.json')
    
        set_tracing_disabled(True)
        #set_trace_processors([LocalTracingProcessor()])
        self.dir_list = [f for f in src_dir.iterdir() if f.is_dir()] 

        client = AsyncOpenAI(base_url=OLLAMA_EP, api_key=OLLAMA_KEY)
        model = OpenAIChatCompletionsModel(model=OLLAMA_MODEL, openai_client=client)
        self.agent = Agent(
            name="Resume Parser Agent",
            instructions="You are an experience HR professional who can parse and analyze resumes to " \
            "extract information about total experience (in years, specific breakup is not required), " \
            "skills and education.",
            model=model
        )

    async def process_file(self, serial: int, file: Path) -> bool:
        subject = file.parent.name
        candidateId = file.name.replace(".md", "")
        prompt = """
        Parse the resume and extract the following information: total experience, skills and education.
        Output should be a JSON object with keys 'total_experience', 'skills' (having list of skills),
        'summary of experience' (list of concised summary of experience mentioned in the resume) 
        and 'education' (list of degrees and additional certifications). Always ensure that the
        output is a valid JSON object and can be parsed without any error. Do not add any additional 
        information in the output apart from the JSON object. If any of the information is not 
        available in the resume, add 'Not Available' as value for that key. Make no assumptions and 
        do not add any information which is not explicitly mentioned in the resume.
        """
        with open(file, 'r') as f:
            resume_content = f.read()
            prompt += f"\n\nResume Content:\n{resume_content}"
        try:
            obj = Query()
            table = self.tinydb.table('candidates')
            srch_result = table.search((obj.candidate == candidateId) & (obj.subject == subject))
            if srch_result:
                #print(f"Record for file {file} in directory {subject} already exists in TinyDB. Skipping processing for this file.")
                return False
            
            print(f"{serial}. Processing file: {file} in directory: {subject} for candidate: {candidateId}")
            response = await Runner.run(self.agent, prompt)
            outputs = response.final_output
            m = re.search(r"```(?:\w*\n)?(.*?)```", outputs, flags=re.S)
            json_outputs = m.group(1) if m else outputs

            json_output = json.loads(json_outputs)
            json_output.update({"subject": subject})
            json_output.update({"candidate": candidateId})
            # Upsert the record based on candidate ID
            #table.upsert(json_output, (obj.candidate == candidateId) & (obj.subject == subject))
            table.insert(json_output)
            print(f"Inserted Record for file {file} in directory {subject} into TinyDB")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for file {file} in directory {subject}: {e}")
        return True

    async def process_directories(self):
        for directory in self.dir_list:
            counter = 1
            for file in directory.iterdir():

                """
                Process only markdown files in the directory.
                Finally write a CSV file in that directory with 
                the summary information of the resumes.
                """
                if file.is_file() and file.suffix == ".md":
                    success = await self.process_file(counter, file)
                    if success:
                        counter += 1

if __name__ == "__main__":
    load_dotenv()

    src_directory = Path("../../data/md/")
    preprocessor = PreprocessResume(src_directory)

    asyncio.run(preprocessor.process_directories())
