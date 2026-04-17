from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class HelloState(TypedDict):
    message: str

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
