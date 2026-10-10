from pydantic import BaseModel, Field
from typing import List, Literal

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(description="Must be exactly 'user' or 'assistant'")
    content: str = Field(min_length=1, max_length=20000, description="Cannot be an empty string and has a token limit safeguard")

class ChatHistory(BaseModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_]+$", description="Username can only contain alphanumeric characters and underscores")
    summary: str = ""
    messages: List[ChatMessage] = []
