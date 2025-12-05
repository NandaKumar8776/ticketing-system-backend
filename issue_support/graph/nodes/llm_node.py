from memory.state import State
from pydantic import BaseModel, Field
from tools.llm_respond import llm_chain_pipeline

# No validation class from pydantic is needed here as input is already validated at Router node for the first message containing the user question

def llm_node(state:State):
    
    """
    LLM Node to generate generic responses using LLM chain pipeline.

    Validation: Not done here as it is already done at Router node.
    Input: current_question from State containing the user's latest question as a string.
    Output: State with 'messages' containing the LLM's response to the question as content and role as assistant. ex. [{'role': 'assistant', 'content': '...LLM response...'}]
    """
    print("-> LLM NODE ->")
    question = state['current_question'][-1]
    print("\nLLM response generating for the question: ", question)
 
    messages= state['messages']
    print("\nMessages (with past history if any) in state for LLM call: ", messages)
   
    # Invoking the pipeline, with output data validation- pydantic
    response = llm_chain_pipeline.invoke(
        {
            "user_query": question,
            "messages": messages,
        }
    )

    print("\nLLM Call's answer to the query: ", response['response']["output"])
 
    # Adding the category to the state's messages
    return {
        "messages": [
            {
                "role": "assistant",
                "content": response['response']["output"]
            }
        ]
    }


