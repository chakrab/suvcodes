import openai
from dotenv import load_dotenv

"""
A basic agent that interacts with a local LLM endpoint to translate text into different language styles.
"""
class BasicLLMAgent:
    def __init__(self):
        load_dotenv()

        self.model_alias = "phi4:14b"
        self.temperature = 0.3
        self.endpoint = "http://localhost:11434/v1"
        self.llm = openai.OpenAI(
            base_url = self.endpoint
        )

    def generate_response(self, language_style: str, user_input: str) -> str:
        prompt = [
        {
            "role": "system",
            "content": f"You are a translator that converts modern English text to {language_style}. Response should be only the translated text. Be concise."
        },
        {
            "role": "user",
            "content": f"Translate the following: '{user_input}'"
        }]
        response = self.llm.chat.completions.create(
            model=self.model_alias,
            temperature=self.temperature,
            messages=prompt,
            stream=False
        )
        return response.choices[0].message.content
    
if __name__ == "__main__":
    agent = BasicLLMAgent()
    prompt = "I am learning to use large language models."
    response = agent.generate_response("Pirate English", prompt)
    print("Response from LLM:", response)