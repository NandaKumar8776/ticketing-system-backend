from memory.state import State
from tools.rag_hybrid_retriever import hybrid_search_rag_pipeline_with_context


# No validation class from pydantic is needed here as input is already validated at Router node for the first message containing the user question

def rag_node(state:State):
    """
    RAG Node to generate responses using Hybrid Search RAG pipeline.
 
    Validation: Not done here as it is already done at Router node.
    Input: current_question from State containing the user's latest question as a string and the entire conversation state.
    Output: State with 'messages' containing the role (assistant), RAG's response to the question, and the context retrieved from RAG. ex. [{'role': 'assistant', 'context': '...retrieved context...'}]
    """

    print("\n-> RAG NODE ->")
    print("\nState received in RAG node:", state)
    question = state["current_question"][-1]
    print("\nHybrid search RAG call initiated for the question: ", question)

    messages= state['messages']
    print("\nMessages (with past history if any) in state for RAG call: ", messages)
   
    # Getting the context from the state field instead of additional_kwargs
    # The Router node stores context in the state's "context" field
    context = state.get("context")
    
    # Extract only page_content from Document objects, discarding metadata
    if context is not None:
        if isinstance(context, list):
            # Extract page_content from each Document object
            extracted_context = []
            for doc in context:
                if hasattr(doc, 'page_content'):
                    # It's a Document object, extract page_content
                    extracted_context.append(doc.page_content)
                elif isinstance(doc, dict) and 'page_content' in doc:
                    # It's a dict with page_content
                    extracted_context.append(doc['page_content'])
                else:
                    # Already a string or other format, use as is
                    extracted_context.append(str(doc) if doc is not None else "")
            context = extracted_context
        elif hasattr(context, 'page_content'):
            # Single Document object
            context = [context.page_content]
        elif isinstance(context, dict) and 'page_content' in context:
            # Single dict with page_content
            context = [context['page_content']]
    
    # Invoking the hybrid search RAG pipeline with context
    response = hybrid_search_rag_pipeline_with_context.invoke(
        {
            "user_query": question,
            "messages": messages,
            "context": context,
        }
    )

    print("\nRag Call's answer to the query: ", response['response']["output"])
    print("\nRag Call's context (retrieved) used to answer the query: ", response['context'])

    # Adding the output to the state's messages (clean message without additional_kwargs)
    # Note: If you need context later, store it in a separate state field instead
    return {
        "messages": [
            {
                "role": "assistant",
                "content": response['response']["output"]
                # Removed "context" to avoid additional_kwargs
            }
        ]
    }


