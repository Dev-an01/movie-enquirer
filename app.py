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
    page_title="Movie Enquirer",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- State Management for Detail View ---
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# --- Custom CSS ---
st.markdown("""
<style>
    /* Subtle enhancements that work reliably in Streamlit */
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1rem; font-weight: 600; padding-bottom: 0.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; color: #059669; }
</style>
""", unsafe_allow_html=True)

# --- Caching Expensive Operations ---
@st.cache_resource(show_spinner=False)
def init_search_engine():
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    return hybrid_search

with st.spinner("🚀 Warming up the search engine..."):
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

# --- Detail View Page ---
if st.session_state.selected_movie:
    movie = st.session_state.selected_movie
    
    st.button("← Back to Search Results", on_click=lambda: st.session_state.update(selected_movie=None))
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if movie.get('img_link'):
            st.image(movie.get('img_link'), use_column_width=True)
        else:
            st.info("No poster available")
            
    with col2:
        st.title(movie.get('title', 'Unknown Title'))
        st.markdown("### Synopsis")
        st.write(movie.get('description', 'No description available.'))
        
        # We can add more metadata here if TMDB dataset provided it
    
    st.stop() # Halt execution so the rest of the search page doesn't render

# --- Sidebar ---
with st.sidebar:
    st.title("🎬 Movie Enquirer")
    st.markdown("A local RAG system built from first principles.")
    
    st.divider()
    
    st.markdown("### 🔗 Links")
    st.markdown("⭐ [GitHub Repository](https://github.com/Dev-an01/movie-enquirer)")
    st.markdown("📖 [Technical Walkthrough](https://github.com/Dev-an01/movie-enquirer/blob/main/blog.md)")
    
    if not client:
        st.warning("⚠️ OPENROUTER_API_KEY is not set. 'Ask Mode' disabled.")

# --- Main Content (Search Page) ---
st.title("🍿 Find Your Next Movie")
st.markdown("Search across 5,000 movies using semantic meaning, exact keywords, or ask our AI for recommendations.")

tab1, tab2 = st.tabs(["🔍 Search Mode", "💬 Ask AI Mode"])

# --- Helper Function for Native Streamlit Results ---
def display_movie_result(idx, doc, rrf_score, tab_key):
    """Renders a single movie result using native Streamlit columns to prevent HTML bugs."""
    with st.container(border=True):
        col1, col2 = st.columns([1, 5])
        
        with col1:
            if doc.get('img_link'):
                st.image(doc.get('img_link'), use_column_width=True)
            else:
                st.caption("No image")
                
        with col2:
            st.subheader(f"#{idx} {doc.get('title', 'Unknown Title')}")
            st.caption(f"✨ Relevance Score: {rrf_score:.4f}")
            
            # Truncate description for the list view
            desc = doc.get('description', '')
            if len(desc) > 200:
                st.write(desc[:200] + "...")
            else:
                st.write(desc)
                
            # Expand button
            st.button("📖 Read More", key=f"btn_{tab_key}_{doc.get('id', idx)}", on_click=lambda d=doc: st.session_state.update(selected_movie=d))

# --- Tab 1: Search Mode ---
with tab1:
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        search_query = st.text_input("Search query", placeholder="e.g., a lonely astronaut trying to survive", label_visibility="collapsed")
    with col_btn:
        search_btn = st.button("Search", use_container_width=True, type="primary")
        
    if search_btn and search_query:
        with st.spinner("🔍 Searching..."):
            results = search_engine.rrf_search(query=search_query, k=60, limit=5)
        
        if results:
            st.success(f"Found {len(results)} matches!")
            for idx, result in enumerate(results, 1):
                display_movie_result(idx, result['document'], result.get('rrf_score', 0), "search")
        else:
            st.info("No results found. Try a different query.")

# --- Tab 2: Ask Mode ---
with tab2:
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        ask_query = st.text_input("Ask for a recommendation", placeholder="e.g., What should I watch if I like mind-bending thrillers?", label_visibility="collapsed")
    with col_btn:
        ask_btn = st.button("Ask AI", use_container_width=True, type="primary")
        
    if ask_btn and ask_query:
        if not client:
            st.error("OpenRouter API key is missing. Please add it to your .env file.")
        else:
            with st.spinner("📚 Retrieving the best movies for context..."):
                results = search_engine.rrf_search(query=ask_query, k=60, limit=5)
            
            if results:
                docs_context = "\n".join(f"- {res['document']['title']}: {res['document']['description']}" for res in results)
                prompt = f"""You are a helpful movie recommendation assistant.
Your task is to provide a natural-language answer to the user's query based on the retrieved movie documents.
Answer questions directly and concisely. Be casual and conversational.

Query: {ask_query}

Retrieved Movies:
{docs_context}

Answer:"""
                
                with st.spinner("🤖 Generating your tailored recommendation..."):
                    try:
                        response = client.chat.completions.create(
                            model="openrouter/free",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        st.info(response.choices[0].message.content)
                        
                        st.markdown("### 📚 Source Movies")
                        for idx, res in enumerate(results, 1):
                            display_movie_result(idx, res['document'], res.get('rrf_score', 0), "ask")
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
            else:
                st.info("No relevant movies found.")
