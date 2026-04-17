from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class HelloState(TypedDict):
    months_owned: int
    defect_category: str
    defect_subcategory: str
    message: str

class HelloCondition:
    def months_owned_check(self, state: HelloState) -> HelloState:
        state['message'] = "ERROR" if state["months_owned"] > 12 else "OK"
        return state

    def defect_category_check(self, state: HelloState) -> HelloState:
        state['message'] = "ERROR" if state["defect_category"] not in ["screen", "power", "camera"] else state["defect_category"].upper()
        return state

    def screen_subcategory_check(self, state: HelloState) -> HelloState:
        state['message'] = "ERROR" if state["defect_subcategory"] not in ["cracked", "flickering"] else "OK"
        return state

    def power_subcategory_check(self, state: HelloState) -> HelloState:
        state['message'] = "ERROR" if state["defect_subcategory"] not in ["not_charging", "overheating"] else "OK"
        return state

    def camera_subcategory_check(self, state: HelloState) -> HelloState:
        state['message'] = "ERROR" if state["defect_subcategory"] not in ["blurry", "not_focusing"] else "OK"
        return state
    
    def warranty_valid(self, state: HelloState) -> HelloState:
        # This is a placeholder for a more complex warranty validation logic
        state['message'] = "Warranty available"
        return state
    
    def warranty_invalid(self, state: HelloState) -> HelloState:
        # This is a placeholder for a more complex warranty validation logic
        state['message'] = "Not under warranty"
        return state

    def route_after_validation(self, state: HelloState) -> str:
        # `validate_currency_node` already set state['message']
        err_code = state['message']
        return err_code   # returns 'ERROR' or 'SUCCESS'

    def create_graph(self):
        graph = StateGraph(HelloState)
        graph.add_node("months_owned_check", self.months_owned_check)
        graph.add_node("defect_category_check", self.defect_category_check)
        graph.add_node("screen_subcategory_check", self.screen_subcategory_check)
        graph.add_node("power_subcategory_check", self.power_subcategory_check)
        graph.add_node("camera_subcategory_check", self.camera_subcategory_check)
        graph.add_node("warranty_valid", self.warranty_valid)
        graph.add_node("warranty_invalid", self.warranty_invalid)

        graph.add_edge(START, "months_owned_check")
        graph.add_conditional_edges(
            "months_owned_check", 
            self.route_after_validation, 
            {
                'ERROR': "warranty_invalid",
                'OK': "defect_category_check",
            }
        )
        graph.add_conditional_edges(
            "defect_category_check", 
            self.route_after_validation, 
            {
                'ERROR': "warranty_invalid",
                'SCREEN': "screen_subcategory_check",
                'POWER': "power_subcategory_check",
                'CAMERA': "camera_subcategory_check"
            }
        )
        graph.add_conditional_edges(
            "screen_subcategory_check", 
            self.route_after_validation, 
            {
                'ERROR': "warranty_invalid",
                'OK': "warranty_valid"
            }
        )
        graph.add_conditional_edges(
            "power_subcategory_check", 
            self.route_after_validation, 
            {
                'ERROR': "warranty_invalid",
                'OK': "warranty_valid"
            }
        )
        graph.add_conditional_edges(
            "camera_subcategory_check", 
            self.route_after_validation, 
            {
                'ERROR': "warranty_invalid",
                'OK': "warranty_valid"
            }
        )
        graph.add_edge("warranty_valid", END)
        graph.add_edge("warranty_invalid", END)
        
        return graph.compile()

if __name__ == "__main__":
    hello_condition = HelloCondition()
    graph = hello_condition.create_graph()
    graph.get_graph().draw_mermaid_png(output_file_path="hello_condition_graph.png")
    result = graph.invoke({ "months_owned": 12, "defect_category": "screen", "defect_subcategory": "cracked", "message": "" })
    print(result)