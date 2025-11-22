from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Defining the state memory for LangGraph 

class State(TypedDict):
    messages: Annotated[list, add_messages]
    