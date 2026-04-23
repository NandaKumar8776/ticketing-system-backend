"""
LangGraph Workflow — Agentic RAG Pipeline

Graph structure:
    Guardrails → (pass) → Router → (conditional) → RAG_call or LLM_call → Evaluator → END
               → (blocked) → END

The Guardrails node runs first, checking for prompt injection, jailbreaks, PII, and
off-topic abuse before any retrieval or LLM call is made. The Router then performs
hybrid retrieval + cross-encoder re-ranking to decide whether to use RAG or a generic
LLM. After the response is generated, the Evaluator scores it using LLM-as-Judge (0-10).
"""

from langgraph.graph import StateGraph, END
from memory.state import State
from graph.nodes.llm_node import llm_node
from graph.nodes.rag_node import rag_node
from graph.nodes.router_node import router_node, route_question
from graph.nodes.evaluator_llm_node import evaluator_node
from graph.nodes.guardrails_node import guardrails_node, route_guardrails

# Langfuse for tracing the workflow — set up in env_setup.py and utils/langfuse.py
from langfuse.langchain import CallbackHandler

workflow = StateGraph(State)

# ── Nodes ──────────────────────────────────────────────────────────
# Guardrails: prompt injection, jailbreak, PII, abuse check — runs before everything
workflow.add_node("Guardrails", guardrails_node)

# Router: Hybrid retrieval scoring + cross-encoder re-ranking → route decision
workflow.add_node("Router", router_node)

# Generation nodes (one of these is selected by the Router)
workflow.add_node("LLM_call", llm_node)
workflow.add_node("RAG_call", rag_node)

# Evaluator: LLM-as-Judge rubric scoring (runs after generation)
workflow.add_node("Evaluator", evaluator_node)

# ── Edges ──────────────────────────────────────────────────────────
# Entry point
workflow.set_entry_point("Guardrails")

# Guardrails → Router (safe) or END (blocked)
workflow.add_conditional_edges(
    "Guardrails",
    route_guardrails,
    {
        "pass": "Router",
        "blocked": END,
    },
)

# Conditional routing: Router → RAG_call or LLM_call
workflow.add_conditional_edges(
    "Router",
    route_question,
    {
        "RAG relevant Issue": "RAG_call",
        "Not Related": "LLM_call",
    },
)

# Both generation nodes → Evaluator → END
workflow.add_edge("RAG_call", "Evaluator")
workflow.add_edge("LLM_call", "Evaluator")
workflow.add_edge("Evaluator", END)

# ── Compile ────────────────────────────────────────────────────────
# Initialize Langfuse CallbackHandler for LangChain (automatic tracing)
langfuse_handler = CallbackHandler()

# Compile the LangGraph app with Langfuse tracing
app = workflow.compile().with_config({"callbacks": [langfuse_handler]})







