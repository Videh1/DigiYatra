import streamlit as st
from core.llm_setup import invoke_travel_agent, generate_master_itinerary
from core.auth import register_user, authenticate_user
from core.db_setup import save_chat_message, get_chat_history, clear_chat_history
from core.memory import run_memory_compression
from core.models import ChatMessage
from core.pdf_gen import generate_itinerary_pdf
from langchain_core.messages import HumanMessage, AIMessage
from core.rag_setup import process_and_embed_document, query_rag_document
import pandas as pd
from geopy.geocoders import Nominatim

@st.cache_data(ttl=3600*24, show_spinner=False)
def get_coordinates(location_name):
    """Fetches GPS coordinates for a given destination and returns a DataFrame for st.map."""
    try:
        geolocator = Nominatim(user_agent="digiyatra_app")
        location = geolocator.geocode(location_name)
        if location:
            return pd.DataFrame({"lat": [location.latitude], "lon": [location.longitude]})
    except:
        pass
    return None

# Configure page settings
st.set_page_config(page_title="DigiYatra - Your AI Travel Companion", page_icon="✈️", layout="wide")

# Initialize session state for authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.logged_in:
    # Hide the "Press Enter to submit form" hint using custom CSS
    st.markdown("""
        <style>
            [data-testid="InputInstructions"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Create empty columns on the sides to center the middle column
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔒 Login to DigiYatra")
        st.markdown("Please log in or register to access your personalized travel assistant.")
        
        # Wrap the forms in a container to act as a "box"
        with st.container(border=True):
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                with st.form("login_form"):
                    login_username = st.text_input("Username")
                    login_password = st.text_input("Password", type="password")
                    submit_login = st.form_submit_button("Login")
                    
                    if submit_login:
                        with st.spinner("Logging in..."):
                            success, msg = authenticate_user(login_username, login_password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = login_username
                            st.session_state.show_login_toast = True
                            st.rerun()
                        else:
                            st.error(msg)
                            
            with tab2:
                with st.form("register_form"):
                    reg_username = st.text_input("New Username")
                    reg_password = st.text_input("New Password", type="password")
                    reg_confirm_password = st.text_input("Confirm Password", type="password")
                    submit_register = st.form_submit_button("Register")
                    
                    if submit_register:
                        if len(reg_username) < 3 or len(reg_password) < 6:
                            st.error("Username must be at least 3 chars and password at least 6 chars.")
                        elif reg_password != reg_confirm_password:
                            st.error("Passwords do not match. Please try again.")
                        else:
                            with st.spinner("Registering..."):
                                success, msg = register_user(reg_username, reg_password)
                            if success:
                                st.success(msg + " Please switch to the Login tab to log in.")
                            else:
                                st.error(msg)
else:
    if st.session_state.get("show_login_toast", False):
        st.toast(f"Login Successful! Welcome {st.session_state.username}.", icon="✅")
        st.session_state.show_login_toast = False

    # Sidebar for Travel Preferences
    with st.sidebar:
        st.title("⚙️ Travel Preferences")
        st.markdown(f"Welcome back, **{st.session_state.username}**!")
        
        app_mode = st.radio("App Mode", ["✈️ AI Travel Agent", "📄 Document Q&A (RAG)"])
        st.divider()
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
            
        if st.button("🗑️ Start New Chat"):
            clear_chat_history(st.session_state.username)
            if "messages" in st.session_state:
                del st.session_state["messages"]
            if "summary" in st.session_state:
                del st.session_state["summary"]
            st.rerun()
            
        # PDF Generation Section
        st.markdown("### 📥 Export Trip Plan")
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            if st.button("⚙️ Compile Final Itinerary"):
                with st.spinner("Synthesizing Master Document..."):
                    chat_history_obj = get_chat_history(st.session_state.username)
                    master_text = generate_master_itinerary(
                        st.session_state.summary, 
                        chat_history_obj
                    )
                    pdf_bytes = generate_itinerary_pdf(
                        st.session_state.username, 
                        master_text
                    )
                    st.session_state["pdf_bytes"] = pdf_bytes
                    
            if "pdf_bytes" in st.session_state:
                st.download_button(
                    label="📥 Download PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name=f"DigiYatra_{st.session_state.username}.pdf",
                    mime="application/pdf"
                )
            
        st.divider()
        st.markdown("Help us tailor recommendations for you!")
        st.session_state.budget = st.selectbox("Budget", ["Budget-friendly", "Mid-Range", "Luxury"])
        st.session_state.travel_style = st.selectbox("Travel Style", ["Solo", "Family", "Couple", "Adventure", "Relaxation"])
        st.session_state.duration = st.slider("Duration (Days)", min_value=1, max_value=7, value=5)
        
        st.divider()
        st.markdown("### About DigiYatra")
        st.markdown("Your smart travel assistant powered by LangChain and Hugging Face.")

    if app_mode == "✈️ AI Travel Agent":
        # Main Application Title
        st.title("✈️ Welcome to DigiYatra")
        st.markdown("I am your personalized AI travel planner. Tell me where you want to go, or ask me to suggest places in a specific state!")

        # Initialize chat history in session state
        if "messages" not in st.session_state:
            chat_history_obj = get_chat_history(st.session_state.username)
            st.session_state.summary = chat_history_obj.summary
            if chat_history_obj.messages:
                st.session_state.messages = chat_history_obj.messages
            else:
                st.session_state.messages = [
                    ChatMessage(role="assistant", content="Hi! I am DigiYatra. Where would you like to visit? You can name a city, state, or even a country!")
                ]

        # Display the conversation summary if it exists
        if st.session_state.summary:
            with st.expander("📝 Current Conversation Summary"):
                st.markdown(st.session_state.summary)

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message.role):
                st.markdown(message.content)

        # Chat input for user
        if prompt := st.chat_input("E.g., I want to visit Kerala... or Suggest places in Rajasthan"):
            # Add user message to chat history
            user_msg = ChatMessage(role="user", content=prompt)
            st.session_state.messages.append(user_msg)
            save_chat_message(st.session_state.username, user_msg)
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Generate Assistant response
            with st.chat_message("assistant"):
                with st.spinner("DigiYatra is planning your trip..."):
                    try:
                        # Convert Pydantic messages to LangChain message objects
                        lc_messages = []
                        for msg in st.session_state.messages[:-1]: # Exclude the current prompt
                            if msg.role == "user":
                                lc_messages.append(HumanMessage(content=msg.content))
                            elif msg.role == "assistant":
                                lc_messages.append(AIMessage(content=msg.content))
                                
                        # Route the prompt and generate response
                        intent, response_stream = invoke_travel_agent(
                            user_input=prompt,
                            budget=st.session_state.budget,
                            travel_style=st.session_state.travel_style,
                            duration=st.session_state.duration,
                            chat_history=lc_messages,
                            summary=st.session_state.summary
                        )
                        
                        response = st.write_stream(response_stream)
                        
                        if intent == "NEW_TRIP":
                            coords = get_coordinates(prompt)
                            if coords is not None:
                                st.map(coords)
                                
                    except Exception as e:
                        response = f"Oops! Something went wrong while connecting to the AI: {e}"
                        st.error(response)
            
            # Add assistant message to chat history
            ast_msg = ChatMessage(role="assistant", content=response)
            st.session_state.messages.append(ast_msg)
            save_chat_message(st.session_state.username, ast_msg)
            
            # Run Memory Compression
            with st.spinner("Compressing memory..."):
                new_summary, compressed_messages = run_memory_compression(
                    st.session_state.username, 
                    st.session_state.summary, 
                    st.session_state.messages
                )
                
                # If compression happened, update state and rerun to reflect UI changes
                if len(compressed_messages) < len(st.session_state.messages):
                    st.session_state.summary = new_summary
                    st.session_state.messages = compressed_messages
                    st.rerun()

    elif app_mode == "📄 Document Q&A (RAG)":
        st.title("📄 Document Q&A (RAG)")
        st.markdown("Upload a travel brochure, itinerary, or policy document, and ask questions about it!")
        
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        
        if uploaded_file:
            # Process document if not already in session state or if file changed
            if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
                with st.spinner("Parsing, Chunking, and Embedding document into ChromaDB..."):
                    try:
                        st.session_state.vectorstore = process_and_embed_document(uploaded_file)
                        st.session_state.current_file = uploaded_file.name
                        st.session_state.rag_messages = [
                            ChatMessage(role="assistant", content="Document embedded successfully! What would you like to know?")
                        ]
                    except ValueError as e:
                        st.error(str(e))
                        st.session_state.current_file = None
            
            # Display RAG Chat History
            if "rag_messages" in st.session_state:
                for message in st.session_state.rag_messages:
                    with st.chat_message(message.role):
                        st.markdown(message.content)
                    
            if prompt := st.chat_input("Ask a question about the document..."):
                user_msg = ChatMessage(role="user", content=prompt)
                st.session_state.rag_messages.append(user_msg)
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving from ChromaDB..."):
                        response = query_rag_document(st.session_state.vectorstore, prompt)
                        st.markdown(response)
                        
                ast_msg = ChatMessage(role="assistant", content=response)
                st.session_state.rag_messages.append(ast_msg)
