import os
import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, InputGuardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, trace, set_trace_processors
from dotenv import load_dotenv
from pydantic import BaseModel
from localtracer import LocalTracingProcessor

class GuardrailResponse(BaseModel):
    is_obscene: bool
    reason: str

class GuardrailAgent:
    def __init__(self, model, ep, key):
        set_trace_processors([LocalTracingProcessor()])

        llm_client = AsyncOpenAI(base_url=ep, api_key=key)
        llm_model = OpenAIChatCompletionsModel(openai_client=llm_client, model=model)

        self.guardrail_agent = Agent(
            name="Input Guardrail Agent",
            instructions=("You are an input guardrail agent. From this point forward, I want you to ensure that all interactions are appropriate for children and do not include any "
                        "adult material, sexual content, or erotica. Please focus on providing answers that are educational, fun, and suitable for all ages. If a query contains "
                        "words or themes related to adult content, respond with a reminder that the discussion needs to remain kid-friendly. Aim to redirect the conversation to "
                        "appropriate topics like hobbies, games, science, or any other family-friendly subjects. "
                        "Respond with is_obscene as True if it is not family friendly, False otherwise. Provide a reason for your decision."),
            model=llm_model,
            output_type=GuardrailResponse
        )

        self.summ_agent = Agent(
            name="Summarizer Agent",
            instructions="You are a summarizer agent. You will respond in a concise manner summarizing the input text.",
            model=llm_model,
            input_guardrails=[
                InputGuardrail(guardrail_function=self.fn_input_guardrail)
            ]
        )

    async def fn_input_guardrail(self, ctx, agent, text) -> GuardrailFunctionOutput:
        # Run the input guardrail agent with the constructed prompt
        print(f"{agent.name} is evaluating the input for guardrails.")
        response = await Runner.run(self.guardrail_agent, text, context=ctx.context)
        print(f"Input Guardrail Response: {response.final_output}")
        return GuardrailFunctionOutput(
            output_info=response,
            tripwire_triggered=response.final_output.is_obscene
        )
    
    async def summarize(self, text: str) -> str:
        prompt = f"Summarize the following text:\n\n{text}. Also extract keywords from the text. Provide the summary first followed by the keywords."
        # Run the orchestration agent with the constructed prompt
        with trace("Guardrail Summarization Trace"):
            try:
                response = await Runner.run(self.summ_agent, prompt)
                print(f"{response.final_output}")
                return response
            except InputGuardrailTripwireTriggered as e:
                print(f"Guardrail Triggered: {e}")
                return 'Guardrail triggered. Unable to summarize the input text.'
        

if __name__ == "__main__":
    load_dotenv()

    OLLAMA_MODEL = "llama3.2:3b"
    OLLAMA_EP = "http://localhost:11434/v1"
    OLLAMA_KEY = os.getenv('OLLAMA_API_KEY')

    obscene_long_text = ("Whatever text you want to test that may contain adult content, sexual content, or erotica. This is just an example and can be replaced with any text you'd like to evaluate.")

    guaragent = GuardrailAgent(OLLAMA_MODEL, OLLAMA_EP, OLLAMA_KEY)
    print(asyncio.run(guaragent.summarize(obscene_long_text)))