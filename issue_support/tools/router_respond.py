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






