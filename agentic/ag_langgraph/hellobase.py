from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class HelloState(TypedDict):
    message: str

"""
A simple example of a state graph that takes a message as input and returns a greeting. The graph 
consists of a single node that modifies the input message to create a greeting. The graph is invoked
with an initial state containing the message "World", and it returns "Hello, World!" as the output.
"""
class HelloBase:
    def create_graph(self):
        graph = StateGraph(HelloState)
        graph.add_node("greet_node", self.greet_node, initial_state={'message': ''})
        graph.add_edge(START, "greet_node")
        graph.add_edge("greet_node", END)
        return graph.compile()

    def greet_node(self, state: HelloState) -> HelloState:
        state['message'] = f'Hello, {state["message"]}!'
        return state

if __name__ == "__main__":
    hello_base = HelloBase()
    graph = hello_base.create_graph()
    graph.get_graph().print_ascii()
    result = graph.invoke({ "message": "World" })
    print(result)
