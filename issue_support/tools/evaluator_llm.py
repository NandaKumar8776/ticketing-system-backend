## PENDING
"""
import os
from utils.helpers import read_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from utils.helpers import llm
from langchain_core.runnables import RunnablePassthrough


class EvaluatorLLMOutput(BaseModel):
    # RUBRIC BASED OUTPUT BY Evaluator LLM FOR LLM RESPONSE EVALUATION
    # output: str= Field(description="The response of the LLM")


evaluator_llm_prompt= read_prompt(os.environ["EVALUATOR_LLM_PROMPT_DIR"])

# Validating the output
parser = JsonOutputParser(pydantic_object= LLMOutput)

prompt = ChatPromptTemplate(
    messages= [
        ("system", llm_prompt),
        ("human", "{user_query}")
    ],
    input_variables= ["user_query"],

    partial_variables= {
        "output_structure": parser.get_format_instructions()
    }
)

# Defining the pipeline
llm_chain_pipeline = (
    {"user_query": RunnablePassthrough()}
    |
    prompt
    |
    llm
    |
    parser
)


####### ROUTER LEGACY CODE - kept for reference

import os
from utils.helpers import read_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from utils.helpers import llm
from langchain_core.runnables import RunnablePassthrough


class RouterOutput(BaseModel):
    category: Literal["PC Issue", "Not Related"]= Field(description="The user's question is categorized into these pre-defined categories. Here, PC Issue is a PC Issue troubleshooting guide. Then, Not Related is anything that is unrelated to PC Issue troubleshooting.")
    reason: str= Field(description="The reason behind choosing the particular category")


supervisor_prompt= read_prompt(os.environ["ROUTER_PROMPT_DIR"])

# Validating the output
parser = JsonOutputParser(pydantic_object= RouterOutput)

prompt = ChatPromptTemplate(
    messages= [
        ("system", supervisor_prompt),
        ("human", "{user_question}")
    ],
    input_variables= ["user_question"],
    partial_variables= {"format_instructions": parser.get_format_instructions()}
)

# Defining the pipeline
router_chain_pipeline = (
    {"user_question": RunnablePassthrough()}
    |
    prompt
    |
    llm
    |
    parser
)







"""


