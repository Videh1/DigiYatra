# Custom Agent Tools

from langchain_community.tools import DuckDuckGoSearchRun
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hotels_from_ddg(query: str) -> str:
    """
    Executes a DuckDuckGo search for the given query and returns raw results.
    If the query is 'NONE', it returns an empty string.
    """
    if query.strip().upper() == "NONE":
        return ""
        
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return results
    except Exception as e:
        return f"Could not fetch hotel data: {str(e)}"
