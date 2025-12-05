from langgraph.graph import StateGraph,END
from memory.state import State
from graph.nodes.llm_node import llm_node
from graph.nodes.rag_node import rag_node
from graph.nodes.router_node import router_node, route_question

# Langfuse for tracing the workflow already set up in env_setup.py and utils/langfuse.py
from langfuse.langchain import CallbackHandler

workflow = StateGraph(State)

# Router (Handles unrelated queries by routing them to a generic LLM response if needed)

workflow.add_node("Router", router_node)

# Adding node functions to the workflow (Each with data validation -in and -out)

workflow.add_node("LLM_call", llm_node)
workflow.add_node("RAG_call", rag_node)


### Setting up the workflow path

# Starting node

workflow.set_entry_point("Router")

# Adding the conditional edges (Router -> LLM, or RAG)

workflow.add_conditional_edges(
    "Router",
    route_question,
    {
        "RAG relevant Issue": "RAG_call",
        "Not Related": "LLM_call",
    }
)

# Adding the edges if any or (All -> END)

# TO-DO Have to add evaluation of output here for RAG with rubric (Subjective EVAL)
workflow.add_edge("RAG_call",END)
workflow.add_edge("LLM_call",END)

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

# Compiling the LangGraph app and adding Langfuse callback for automatic tracing

app= workflow.compile().with_config({"callbacks": [langfuse_handler]})







