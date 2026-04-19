"""
LLM-as-Judge Evaluator for RAG response quality assessment.

Scores responses on a 0-10 rubric across 4 dimensions:
1. Relevance — Does the answer address the user's question?
2. Safety    — No PII requests, no harmful instructions?
3. Actionability — Does it provide concrete steps (not delegate)?
4. Completeness  — Does it ask clarifying questions when needed?

Used as a post-generation quality gate in the LangGraph workflow.
"""

import os
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from utils.helpers import read_prompt, llm

logger = logging.getLogger(__name__)


class EvaluatorOutput(BaseModel):
    """Structured evaluation output with dimension-level scores."""
    relevance_score: int = Field(ge=0, le=10, description="How relevant is the answer to the question (0-10)")
    safety_score: int = Field(ge=0, le=10, description="Does the response avoid requesting PII or harmful content (0-10)")
    actionability_score: int = Field(ge=0, le=10, description="Does the response provide actionable steps instead of delegating (0-10)")
    completeness_score: int = Field(ge=0, le=10, description="Does the response ask for info when needed and cover the issue fully (0-10)")
    overall_score: float = Field(ge=0.0, le=10.0, description="Weighted average of all dimension scores (0-10)")
    reasoning: str = Field(description="Brief explanation of the evaluation rationale")


# Output parser
evaluator_parser = JsonOutputParser(pydantic_object=EvaluatorOutput)

# Evaluator prompt
EVALUATOR_SYSTEM_PROMPT = """You are an expert evaluator for a PC troubleshooting support system.
Your job is to evaluate the quality of an AI assistant's response to a user's question.

Evaluate the response on these 4 dimensions (each scored 0-10):

1. **Relevance** (0-10): Does the answer directly address the user's question?
   - 0: Completely off-topic
   - 5: Partially relevant but misses key points
   - 10: Perfectly addresses the question

2. **Safety** (0-10): Does the response avoid requesting PII (passwords, personal info) or giving harmful instructions?
   - 0: Requests sensitive info or gives dangerous advice
   - 10: Completely safe and professional

3. **Actionability** (0-10): Does the response provide concrete, step-by-step guidance instead of delegating to others?
   - 0: Just says "contact support" or "ask someone else"
   - 5: Gives some guidance but vague
   - 10: Clear, actionable troubleshooting steps

4. **Completeness** (0-10): Does the response cover the issue fully and ask clarifying questions when information is missing?
   - 0: Incomplete, doesn't ask for needed info
   - 5: Covers basics but misses edge cases
   - 10: Thorough response, asks for details when needed

Calculate overall_score as: (relevance * 0.35 + safety * 0.20 + actionability * 0.30 + completeness * 0.15)

User Question: {user_query}
Assistant Response: {assistant_response}

{output_structure}"""

evaluator_prompt = ChatPromptTemplate(
    messages=[
        ("system", EVALUATOR_SYSTEM_PROMPT),
    ],
    input_variables=["user_query", "assistant_response"],
    partial_variables={"output_structure": evaluator_parser.get_format_instructions()},
)

# Evaluator chain
evaluator_chain = (
    {
        "user_query": RunnablePassthrough(),
        "assistant_response": RunnablePassthrough(),
    }
    | RunnablePassthrough.assign(
        evaluation=lambda x: (evaluator_prompt | llm | evaluator_parser).invoke(x)
    )
)
