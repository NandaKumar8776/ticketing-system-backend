# LLM Setup

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
)

# Text file reading

def read_prompt(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
        return content

# LLM Output formatter to return only the last message's content

def output_formatter(state):
    return state['messages'][-1].content


# DOC Output formatter

def doc_output_formatter(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Embeddings

from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

