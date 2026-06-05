# LANGGRAPH

# Projects
## hellobase
A simple example of a state graph that takes a message as input and returns a greeting. The graph 
consists of a single node that modifies the input message to create a greeting. The graph is invoked
with an initial state containing the message "World", and it returns "Hello, World!" as the output.

## hellocondition
This code defines a HelloCondition class that implements a state graph for validating warranty claims based 
on the number of months owned, defect category, and defect subcategory. The graph checks if the device is 
still under warranty and routes to appropriate nodes based on the validation results. The final output indicates
whether the warranty is valid or not.