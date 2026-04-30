"""
Evaluator LLM Node — LLM-as-Judge quality gate.

Runs after the RAG or LLM node to score the generated response on a rubric.
Scores are stored in state for API response and Langfuse logging.
"""

import logging
from memory.state import State
from tools.evaluator_llm import evaluator_chain

logger = logging.getLogger(__name__)


def evaluator_node(state: State):
    """
    Evaluate the quality of the last assistant response using LLM-as-Judge.

    Scores the response on 4 dimensions (relevance, safety, actionability,
    completeness) and computes a weighted overall score (0-10).

    Args:
        state (State): Current state containing:
            - current_question: List with the user's question
            - messages: Conversation history (last message is the assistant's response)

    Returns:
        dict: State update with:
            - eval_score: Overall rubric score (0-10 float)
    """
    print("-> EVALUATOR NODE ->")

    question = state["current_question"][-1]
    last_message = state["messages"][-1]

    # Extract assistant response text
    assistant_response = getattr(last_message, "content", "")
    if not assistant_response and isinstance(last_message, dict):
        assistant_response = last_message.get("content", "")

    if not assistant_response:
        # An empty response means an upstream node failed silently — flag it.
        logger.error(
            "Evaluator received an empty assistant response. "
            "This likely means a generation node failed without raising an exception."
        )
        return {"eval_score": None}

    try:
        result = evaluator_chain.invoke({
            "user_query": question,
            "assistant_response": assistant_response,
        })

        evaluation = result.get("evaluation", {})
        overall_score = evaluation.get("overall_score", None)
        reasoning = evaluation.get("reasoning", "")

        logger.info(
            f"Evaluation complete — overall: {overall_score}/10 | "
            f"relevance: {evaluation.get('relevance_score')} | "
            f"safety: {evaluation.get('safety_score')} | "
            f"actionability: {evaluation.get('actionability_score')} | "
            f"completeness: {evaluation.get('completeness_score')}"
        )
        logger.info(f"Evaluator reasoning: {reasoning}")

        return {"eval_score": overall_score}

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return {"eval_score": None}