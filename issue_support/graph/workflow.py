from langgraph.graph import StateGraph,END
from memory.state import State
from graph.nodes.llm_node import llm_node

# Langfuse for tracing the workflow already set up in env_setup.py and utils/langfuse.py
from langfuse.langchain import CallbackHandler

workflow = StateGraph(State)

# Adding node functions to the workflow (Each with data validation -in and -out)

workflow.add_node("LLM_call", llm_node)

# Starting node

workflow.set_entry_point("LLM_call")

# Adding the edges if any or (All -> END)

workflow.add_edge("LLM_call",END)

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

# Compiling the LangGraph app and adding Langfuse callback for automatic tracing

app= workflow.compile().with_config({"callbacks": [langfuse_handler]})







