from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

"""
This module defines a wrapper class for the OllamaLLM, which is a language model that can be used to
generate responses to prompts. The wrapper class provides a method to generate responses based on a 
given prompt, using a predefined chat prompt template. The main function demonstrates how to use the 
wrapper class to generate a response to a specific question.
"""
class OllamaLLMWrapper():
    def __init__(self):
        self.ollama_llm = OllamaLLM(model="command-r7b:7b")

    def generate_response(self, prompt: str) -> str:
        chat_prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}.")
        print(f"Generated prompt {chat_prompt}")
        # Create a chain that combines the chat prompt and the Ollama LLM
        chain = chat_prompt | self.ollama_llm
        response = chain.invoke({"question": prompt})
        return response
    
if __name__ == "__main__":
    ollama_wrapper = OllamaLLMWrapper()
    prompt = "What is the capital of UAE? Respond with only the name of the city."
    response = ollama_wrapper.generate_response(prompt)
    print(response)