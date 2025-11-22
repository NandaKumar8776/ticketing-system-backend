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
