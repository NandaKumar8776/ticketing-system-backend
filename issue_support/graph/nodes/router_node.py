from memory.state import State
from pydantic import BaseModel, Field
from tools.router_respond import router_chain_pipeline

class RouterInput(BaseModel):
    user_question: str= Field(description="This is the user's question, that needs to be categorized")


def router_node(state: State):
    """
    Router Node to categorize user questions into predefined categories using LLM chain pipeline.

    Possible Categories:
    1. "PC Issue" - The user's question is related to PC Issue troubleshooting.
    2. "Not Related" - The user's question is unrelated to PC Issue troubleshooting.

    Validation: The input question is validated using pydantic to ensure it is a string.
    Input: State with 'messages' containing the user's question as content and role as the last message. ex. [{'role': 'user', 'content': 'Hiii'}]
    Output: State dict with updated messages
    """

    print("-> ROUTER ->")
    
    last_msg = state["messages"][-1]
    print("Last message in state:", last_msg)
    
    # Extract text content from the last message which is the question
    question = getattr(last_msg, 'content', last_msg)

    # Validating the input question
    valiadted_question = RouterInput(user_question= question)
    valiadted_question = valiadted_question.user_question

    print(f"\nRouter Node processing the question [{valiadted_question}] for routing")
    

    # Invoking the router pipeline - output validated with pydantic
    response = router_chain_pipeline.invoke(valiadted_question)

    print("\nRouter Node categorized the question to: ", response)

    # Return state dict with routing decision added to messages
    return {
        "messages": [
            {
                "role": "ai",
                "content": response['reason'],
                "category": response['category']
            }
        ],
        
        "current_question": valiadted_question
    }


def route_question(state: State) -> str:
    """
    Routing function that determines which path the question should take.
    
    Returns:
        str: Either "PC Issue" or "Not Related" to route to appropriate node
    """
    last_msg = state["messages"][-1]
    
    # Extract text content from the last message
    question = getattr(last_msg, 'content', last_msg)

    # Validating the input question
    valiadted_question = RouterInput(user_question= question)
    valiadted_question = valiadted_question.user_question

    # Invoking the router pipeline
    response = router_chain_pipeline.invoke(valiadted_question)
    
    # Return only the category for routing
    return response['category']