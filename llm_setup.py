import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (like HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

def get_llm():
    """Initializes and returns the HuggingFace Chat LLM."""
    
    # MUST BE "conversational" FOR THIS MODEL ON HUGGINGFACE FREE API
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="conversational", 
        max_new_tokens=512,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    chat_model = ChatHuggingFace(llm=llm)
    
    return chat_model

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnableLambda
from core.tools import fetch_hotels_from_ddg

def invoke_travel_agent(user_input, budget, travel_style, duration, chat_history, summary):
    """Routes the user input to either the Destination Chain or Conversational Chain."""
    llm = get_llm()
    parser = StrOutputParser()

    # 1. Router Chain
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict intent classifier. Given the chat history and the user's latest input, classify the intent into one of two categories:\n1. 'NEW_TRIP' - The user is providing a destination (like a city or state) to start planning a new trip. Even if they are just replying with a single word (e.g. 'Indore') to your greeting, it is a NEW_TRIP.\n2. 'FOLLOW_UP' - The user is asking follow-up questions or discussing details about a destination you are ALREADY talking about.\n\nOutput EXACTLY one of these two words: NEW_TRIP or FOLLOW_UP. No punctuation.\n\nConversation Summary so far: {summary}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_input}")
    ])
    router_chain = router_prompt | llm | parser
    
    # Classify the intent
    intent_raw = router_chain.invoke({
        "user_input": user_input,
        "chat_history": chat_history,
        "summary": summary
    })
    intent = intent_raw.strip().upper()
    
    # Safety fallback
    if "NEW_TRIP" not in intent and "FOLLOW_UP" not in intent:
        intent = "FOLLOW_UP"
    else:
        intent = "NEW_TRIP" if "NEW_TRIP" in intent else "FOLLOW_UP"

    # 2. Itinerary Chain (Process A)
    itinerary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are DigiYatra, a friendly AI travel planner.
        
The user is planning a NEW trip. Their preferences are:
- Budget: {budget}
- Travel Style: {travel_style}
- Duration: {duration} days

Conversation Summary so far: {summary}

Verify if their destination is a real place. If it's gibberish, politely ask for a real place.
If it IS a real place, provide a tailored recommendation including an overview, 3-5 activities matching their budget/style, and a travel tip. DO NOT suggest hotels."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_input}")
    ])
    itinerary_chain = itinerary_prompt | llm | parser

    # 3. Hotel Agent Pipeline (Process B)
    # 3a. Query Extractor
    hotel_query_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a smart location analyzer. 
The user wants to travel to a destination with a {budget} budget.
If the user's destination is a broad region (State, Province, Country) like "Gujarat" or "India", output EXACTLY the word: NONE
If the user's destination is a specific city, town, or exact landmark (like "Ahmedabad" or "Sabarmati Ashram"), generate a search engine query to find hotels there.
Example Output: Top {budget} hotels in Ahmedabad"""),
        ("user", "{user_input}")
    ])
    query_chain = hotel_query_prompt | llm | parser

    # 3b. Search Execution
    ddg_search = RunnableLambda(fetch_hotels_from_ddg)

    # 3c. Formatter
    hotel_format_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Hotel Recommendation AI. 
Here are raw internet search results for hotels matching the user's budget:
{search_results}

If the search results are empty or say "NONE", reply EXACTLY with: "*Please specify a city if you would like hotel recommendations.*"
Otherwise, read the raw search results and format them into a beautiful list of 2-3 specific hotels. Include the name and a 1-sentence description. Do not hallucinate prices. Output ONLY the hotel recommendations.""")
    ])
    format_chain = hotel_format_prompt | llm | parser

    # 3d. Connect Process B
    hotel_agent_chain = query_chain | ddg_search | (lambda res: {"search_results": res}) | format_chain

    # 4. RunnableParallel Merge
    destination_parallel_chain = RunnableParallel(
        itinerary=itinerary_chain,
        hotels=hotel_agent_chain
    )

    # 3. Conversational Chain
    conv_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are DigiYatra, a friendly AI travel assistant.
        
The user is asking a follow-up question about their ongoing trip discussion.
Conversation Summary so far: {summary}

Answer conversationally based on the summary and chat history. Do not generate a massive new itinerary unless explicitly asked."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_input}")
    ])
    conv_chain = conv_prompt | llm | parser

    # 5. Routing Execution
    if intent == "NEW_TRIP":
        def generate_new_trip():
            inputs = {
                "user_input": user_input,
                "budget": budget,
                "travel_style": travel_style,
                "duration": duration,
                "chat_history": chat_history,
                "summary": summary
            }
            # Stream the itinerary first
            for chunk in itinerary_chain.stream(inputs):
                yield chunk
            
            yield "\n\n### 🏨 Accommodations\n"
            
            # Then stream the hotel recommendations
            for chunk in hotel_agent_chain.stream(inputs):
                yield chunk
                
        return intent, generate_new_trip()
    else:
        def generate_follow_up():
            inputs = {
                "user_input": user_input,
                "chat_history": chat_history,
                "summary": summary
            }
            for chunk in conv_chain.stream(inputs):
                yield chunk
                
        return intent, generate_follow_up()

def generate_master_itinerary(summary: str, chat_history: list) -> str:
    """Uses the LLM to synthesize the conversation memory into a single Master Itinerary."""
    llm = get_llm()
    parser = StrOutputParser()
    
    pdf_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional travel document formatter.
Review the entire conversation summary and the recent chat history provided below.
Your task is to synthesize all agreed-upon details into a single, highly-organized Master Itinerary.
It must include an Overview, Confirmed Hotels, Daily Activities, and Travel Tips.

CRITICAL INSTRUCTION: ONLY include hotels, flights, and activities that were EXPLICITLY discussed and agreed upon in the provided memory or chat history. DO NOT invent or hallucinate new details. If a hotel or activity was not mentioned, state that it has not been decided yet.

Format your response in clean Markdown (use ## for headers, ** for bold, and - for bullets). Do not use complex HTML or emojis.

Conversation Summary: {summary}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Please generate the final Master Itinerary for my trip based on our conversation.")
    ])
    
    pdf_chain = pdf_prompt | llm | parser
    
    return pdf_chain.invoke({
        "summary": summary,
        "chat_history": chat_history
    })
