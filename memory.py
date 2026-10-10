from core.llm_setup import get_llm
from core.db_setup import update_chat_history_and_summary
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.models import ChatMessage
from typing import List

def run_memory_compression(username: str, current_summary: str, messages: List[ChatMessage], max_messages: int = 6):
    """
    Checks if the raw messages exceed the limit. If so, condenses the oldest ones into the summary.
    Returns (new_summary, remaining_messages).
    """
    if len(messages) <= max_messages:
        return current_summary, messages
        
    # We will keep the 4 most recent messages (2 user/assistant pairs)
    keep_count = 4
    
    messages_to_compress = messages[:-keep_count]
    messages_to_keep = messages[-keep_count:]
    
    # Format the messages to compress into a string
    conversation_str = ""
    for msg in messages_to_compress:
        role = "User" if msg.role == "user" else "AI"
        conversation_str += f"{role}: {msg.content}\n"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant tasked with condensing conversation history. Summarize the following conversation log. If there is an Existing Summary, combine it seamlessly with the new information. Keep it concise but retain important facts, preferences, and locations mentioned. The final summary must NOT exceed 2000 words. If it gets too long, prioritize the most important details and discard older, less relevant information. Output ONLY the new summary text."),
        ("user", "Existing Summary: {current_summary}\n\nNew Conversation to Compress:\n{conversation_str}")
    ])
    
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    
    new_summary = chain.invoke({
        "current_summary": current_summary if current_summary else "No existing summary.",
        "conversation_str": conversation_str
    })
    
    # Save to database
    update_chat_history_and_summary(username, new_summary, messages_to_keep)
    
    return new_summary, messages_to_keep
