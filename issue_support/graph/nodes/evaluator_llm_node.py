## PENDING

from memory.state import State
from tools.llm_respond import llm_chain_pipeline


def evaluator_llm_node(state:State):
    
    # Validation already done at Supervisor node
    question = state['messages'][0]
    # Extract text whether message is a dict, object with .content, or raw string
    question_text = question.get('content', question)

    print("\nEvaluator checking the response of the previous LLM call")

    # Invoking the pipeline, with data validation- pydantic
    response = llm_chain_pipeline.invoke(question_text)

    print("\nLLM response: ", response['output'])

    # Adding the category to the state's messages
    return {"messages": response['output']}