import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

try:
    print("Initializing LLM...")
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="conversational",
        max_new_tokens=512,
        temperature=0.7
    )
    chat_model = ChatHuggingFace(llm=llm)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant."),
        ("user", "Hello {name}")
    ])
    
    chain = prompt | chat_model | StrOutputParser()
    
    print("Invoking chain...")
    res = chain.invoke({"name": "John"})
    print("RESPONSE:", res)
except Exception as e:
    print(f"ERROR: {e}")
