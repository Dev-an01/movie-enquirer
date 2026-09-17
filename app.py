import os
import sys
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Ensure the cli directory is in the path to import lib modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'cli'))

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Enquirer Showcase",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Enquirer")
st.markdown("""
Welcome to the **Movie Enquirer**! This is a plain showcase of a local Retrieval-Augmented Generation (RAG) system built from first principles.
[Read the technical walkthrough in blog.md](https://github.com/Dev-an01/movie-enquirer/blob/main/blog.md)
""")

# --- Caching Expensive Operations ---
@st.cache_resource(show_spinner=False)
def init_search_engine():
    with st.spinner("Initializing the search engine (loading models and cache)..."):
        documents = load_movies()
        hybrid_search = HybridSearch(documents)
        return hybrid_search

try:
    search_engine = init_search_engine()
except Exception as e:
    st.error(f"Failed to load search engine: {e}")
    st.stop()

# --- OpenRouter Client ---
api_key = os.getenv("OPENROUTER_API_KEY")
client = None
if api_key:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
else:
    st.warning("⚠️ OPENROUTER_API_KEY is not set. 'Ask Mode' will be disabled.")

# --- Tabs ---
tab1, tab2 = st.tabs(["🔍 Search Mode", "💬 Ask Mode"])

# --- Tab 1: Search Mode ---
with tab1:
    st.header("Search Movies")
    st.markdown("Query the local lexical and semantic index. No LLM calls are made here.")
    
    search_query = st.text_input("Enter a search query (e.g., 'a lonely astronaut trying to survive')", key="search_query")
    if st.button("Search", key="search_button"):
        if search_query:
            with st.spinner("Searching..."):
                results = search_engine.rrf_search(query=search_query, k=60, limit=5)
            
            if results:
                for idx, result in enumerate(results, 1):
                    doc = result['document']
                    with st.expander(f"**{idx}. {doc.get('title', 'Unknown Title')}** (Score: {result.get('score', 0):.4f})"):
                        st.write(doc.get('description', 'No description available.'))
            else:
                st.info("No results found.")
        else:
            st.warning("Please enter a query.")

# --- Tab 2: Ask Mode ---
with tab2:
    st.header("Ask About Movies")
    st.markdown("Generates an answer using OpenRouter, grounded in the retrieved sources.")
    
    ask_query = st.text_input("Ask a question (e.g., 'What should I watch if I like mind-bending thrillers?')", key="ask_query")
    
    if st.button("Ask", key="ask_button"):
        if not client:
            st.error("OpenRouter API key is missing. Please add it to your .env file.")
        elif ask_query:
            with st.spinner("Retrieving sources..."):
                results = search_engine.rrf_search(query=ask_query, k=60, limit=5)
            
            if results:
                # Format retrieved documents for the prompt
                docs_context = "\n".join(
                    f"- {res['document']['title']}: {res['document']['description']}"
                    for res in results
                )
                
                prompt = f"""You are a helpful movie recommendation assistant.
Your task is to provide a natural-language answer to the user's query based on the retrieved movie documents.
Answer questions directly and concisely. Be casual and conversational.

Query: {ask_query}

Retrieved Movies:
{docs_context}

Answer:"""
                
                with st.spinner("Generating answer..."):
                    try:
                        response = client.chat.completions.create(
                            model="openrouter/free",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        answer = response.choices[0].message.content
                        st.success("Here's what I found:")
                        st.write(answer)
                        
                        st.markdown("### Sources")
                        for idx, res in enumerate(results, 1):
                            st.write(f"{idx}. **{res['document']['title']}**")
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
            else:
                st.info("No relevant movies found to answer your question.")
        else:
            st.warning("Please enter a question.")
