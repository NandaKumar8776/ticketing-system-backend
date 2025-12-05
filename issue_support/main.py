import config.env_setup
from graph.workflow import app


def chatbot():
    """
    Interactive command-line chatbot with continuous conversation ability.
    
    This function runs an interactive loop that:
    1. Accepts user input
    2. Invokes the LangGraph workflow to process the query
    3. Displays the assistant's response
    4. Maintains conversation history across turns
    
    The chatbot routes queries to either RAG (for RAG relevant Issues) or generic LLM
    (for unrelated questions) based on relevance scoring.
    
    Commands:
        - 'exit', 'quit', 'bye', 'goodbye': End the conversation
        - Ctrl+C: Interrupt and exit
    
    Returns:
        None
    """
    print("\n" + "="*60)
    print("TICKETING SYSTEM CHATBOT")
    print("="*60)
    print("Welcome! I'm here to help with RAG relevant Issues or general questions.")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    messages = []
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\nChatbot: Thank you for using the Ticketing System. Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input:
                print("Chatbot: Please enter a message.\n")
                continue
            
            # Add user message to conversation history
            messages.append({"role": "user", "content": user_input})
            
            # Create state with full conversation history
            state = {"messages": messages}
            
            # Invoke the workflow
            result = app.invoke(state)
            
            # Extract the assistant's response from the result
            if result.get("messages"):
                last_message = result["messages"][-1]
                
                # Handle both LangChain message objects and dicts
                if hasattr(last_message, 'content'):
                    assistant_response = last_message.content
                else:
                    assistant_response = last_message.get("content", "Sorry, I couldn't generate a response.")
                
                print(f"\nChatbot: {assistant_response}\n")
                
                # Add assistant response to conversation history
                messages.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
            else:
                print("Chatbot: Sorry, I couldn't process your request. Please try again.\n")
                
        except KeyboardInterrupt:
            print("\n\nChatbot: Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nChatbot: An error occurred: {str(e)}")
            print("Please try again.\n")


if __name__ == "__main__":
    chatbot()