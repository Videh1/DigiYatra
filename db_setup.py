import os
# pyrefly: ignore [missing-import]
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from core.models import ChatMessage, ChatHistory
from typing import List

def get_db():
    """Initializes and returns the MongoDB database instance."""
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable not found. Please set it in .env")
        
    client = MongoClient(mongo_uri)
    # Return the 'digiyatra' database
    db = client["digiyatra"]
    return db

def save_chat_message(username: str, message: ChatMessage):
    """Saves a chat message for a user."""
    db = get_db()
    chat_collection = db["chat_history"]
    
    chat_collection.update_one(
        {"username": username},
        {"$push": {"messages": message.model_dump()}},
        upsert=True
    )

def get_chat_history(username: str) -> ChatHistory:
    """Retrieves the chat history and summary for a user."""
    db = get_db()
    chat_collection = db["chat_history"]
    
    document = chat_collection.find_one({"username": username})
    if document:
        messages = [ChatMessage(**msg) for msg in document.get("messages", [])]
        return ChatHistory(username=username, summary=document.get("summary", ""), messages=messages)
    
    return ChatHistory(username=username, summary="", messages=[])

def update_chat_history_and_summary(username: str, new_summary: str, remaining_messages: List[ChatMessage]):
    """Updates the conversation summary and overwrites the messages array."""
    db = get_db()
    chat_collection = db["chat_history"]
    
    messages_dicts = [msg.model_dump() for msg in remaining_messages]
    
    chat_collection.update_one(
        {"username": username},
        {"$set": {
            "summary": new_summary,
            "messages": messages_dicts
        }},
        upsert=True
    )

def clear_chat_history(username: str):
    """Clears the chat history and summary for a user in the database."""
    db = get_db()
    chat_collection = db["chat_history"]
    chat_collection.update_one(
        {"username": username},
        {"$set": {"messages": [], "summary": ""}}
    )
