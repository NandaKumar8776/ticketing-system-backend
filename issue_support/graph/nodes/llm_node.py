from memory.state import State
from tools.llm_respond import llm_chain_pipeline


def llm_node(state:State):
    
    # Validation already done at Supervisor node
    question = state['messages'][0]
    # Always extract .content if present
    question = getattr(question, 'content', question)

    print("\nLLM response generating")

    # Invoking the pipeline, with data validation- pydantic
    response = llm_chain_pipeline.invoke(question)

    print("\nLLM response: ", response['output'])

    # Adding the category to the state's messages
    return {"messages": [response['output']]}


