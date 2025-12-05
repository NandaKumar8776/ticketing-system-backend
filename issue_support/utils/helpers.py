# LLM Setup

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
)

rag_llm = ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905"
)
# Text file reading

def read_prompt(filepath):
    """
    Read prompt text from a file.
    
    Args:
        filepath (str): Path to the prompt file to read.
    
    Returns:
        str: The content of the prompt file.
    """
    with open(filepath, 'r') as file:
        content = file.read()
        return content

# LLM Output formatter to return only the last message's content

def output_formatter(state):
    """
    Extract the content from the last message in the state.
    
    Used to format the final output from the LangGraph workflow for display.
    
    Args:
        state (dict): The state dictionary containing a 'messages' list.
    
    Returns:
        str: The content of the last message in the messages list.
    """
    return state['messages'][-1].content


# DOC Output formatter

def doc_output_formatter(docs):
    """Format documents into a context string by joining their page_content."""
    try:
        formatted_parts = []
        for i, doc in enumerate(docs):
            try:
                # Handle both Document objects and dicts
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif isinstance(doc, dict) and 'page_content' in doc:
                    content = doc['page_content']
                else:
                    content = str(doc)
                
                # Ensure content is a string
                if content is None:
                    content = ""
                else:
                    content = str(content)
                
                formatted_parts.append(content)
            except Exception as e:
                print(f"[doc_formatter] Error processing doc {i}: {e}")
                continue
        
        return "\n\n".join(formatted_parts)
    except Exception as e:
        print(f"[doc_formatter] Error in doc_output_formatter: {e}")
        import traceback
        print(traceback.format_exc())
        return ""

# Embeddings

from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

