from memory.state import State
from pydantic import BaseModel, Field
from tools.llm_respond import llm_chain_pipeline

# No validation class from pydantic is needed here as input is already validated at Router node for the first message containing the user question

def llm_node(state:State):
    """
    LLM Node to generate generic responses using LLM chain pipeline.
    
    This node handles questions that are not related to RAG document. It uses the LLM
    to generate a generic response without RAG context.

    Validation: Not done here as it is already done at Router node.
    
    Args:
        state (State): The current state containing:
            - current_question: List with the latest user question as a string
            - messages: List of conversation history messages
    
    Returns:
        dict: State update with 'messages' containing:
            - role: "assistant"
            - content: LLM's response to the question (clean message without additional_kwargs)
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


