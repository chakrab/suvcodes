# OPENAI

## OpenAI Agents

## MLFlow
On MacOS, Control Center runs on 5000 and 7000. So, we have to run mlflow on a different port.

  uvx mlflow server -p 5050

For this set of projects, we are not using MLFlow. We have our own custom hook implementation.

# Projects
## firstagent.py
This class defines a simple agent that translates English text to Spanish using an OpenAI model. The agent is initialized with the model name, endpoint, and API key. The translate method constructs a prompt and runs the agent to get the Spanish translation of the input English text.

## guardrails.py
This class defines a GuardrailAgent that uses two agents: one for evaluating the input text against guardrails and another for summarizing the text. The input guardrail agent checks if the text contains any adult content, sexual content, or erotica and responds accordingly. The summarizer agent provides a concise summary of the input text while ensuring that it adheres to the guardrails set by the first agent. If the guardrail is triggered, it will return a message indicating that the input text cannot be summarized due to inappropriate content.

## handofflocal.py
This class defines a local agent that can hand off user requests to either a Poet Agent or a Short Story Agent based on the content of the request. It uses the OpenAI API for processing the requests and generating responses. The agent is designed to run locally, and it includes tracing capabilities for debugging and analysis purposes.

## localtracer.py
A tracing processor that builds a JSON flow. Traces contain spans (one trace -> many spans).On trace end the trace is serialized to JSON (printed or can be saved).

## orchestrator.py
The Orchestrator class initializes multiple agents, each specializing in a different domain (Chemistry, History, Astronomy), and a main driver agent that orchestrates the use of these tools to answer user questions. The driver agent determines which specialized agent to use based on the user's request and retrieves the appropriate information.

## tooluse.py
The ToolUsage class demonstrates how to use a tool within an agent to fetch Chinese Zodiac personality traits based on a birth year. It initializes an agent with the necessary model and tools, and provides a method to run the agent with a given birth year.

This calculation is based on the traditional Chinese Zodiac system, which assigns an animal and an element to each year in a 12-year cycle. The personality traits are derived from the assigned animal and element. However, calculating with only the birth year may not be entirely accurate for determining the Chinese Zodiac sign, as the Chinese do not use a solar calendar, but instead uses a lunar calendar. 

## contextmanager.py
We will build a Travel Recommendation System that provides personalized travel suggestions based on user preferences and current location. The system will utilize a Context Manager to manage user preferences and current context, allowing it to generate tailored travel recommendations.
"""