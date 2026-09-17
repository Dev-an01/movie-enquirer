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

# --- Custom CSS for Impeccable Design ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: #f9fafb;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #2563eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .movie-title {
        font-size: 1.25rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .movie-score {
        font-size: 0.9rem;
        color: #059669;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .movie-desc {
        font-size: 1rem;
        color: #374151;
        line-height: 1.5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3163/3163478.png", width=100)
    st.markdown("### 🎬 Movie Enquirer")
    st.markdown("A local Retrieval-Augmented Generation (RAG) system built from first principles.")
    
    st.divider()
    
    st.markdown("### 🔗 Links")
    st.markdown("⭐ [GitHub Repository](https://github.com/Dev-an01/movie-enquirer)")
    st.markdown("📖 [Read the Technical Walkthrough](https://github.com/Dev-an01/movie-enquirer/blob/main/blog.md)")
    
    st.divider()
    st.markdown("<small>Built with Python, Streamlit, BM25, and MiniLM.</small>", unsafe_allow_html=True)

# --- Main Content ---
st.markdown('<p class="main-header">🍿 Find Your Next Movie</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Search across 5,000 movies using semantic meaning, exact keywords, or ask our AI for recommendations.</p>', unsafe_allow_html=True)

# --- Caching Expensive Operations ---
@st.cache_resource(show_spinner=False)
def init_search_engine():
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    return hybrid_search

with st.spinner("🚀 Warming up the search engine (loading models and cache)..."):
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
    st.sidebar.warning("⚠️ OPENROUTER_API_KEY is not set. 'Ask Mode' disabled.")

# --- Tabs ---
tab1, tab2 = st.tabs(["🔍 Search Mode", "💬 Ask AI Mode"])

# --- Helper Function for Results ---
def display_movie_card(idx, title, desc, rrf_score):
    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">#{idx} {title}</div>
        <div class="movie-score">✨ Relevance Score: {rrf_score:.4f}</div>
        <div class="movie-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Tab 1: Search Mode ---
with tab1:
    st.markdown("#### Query the local lexical and semantic index.")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input("What kind of movie are you looking for?", placeholder="e.g., a lonely astronaut trying to survive", label_visibility="collapsed")
    with col2:
        search_btn = st.button("Search Movies", use_container_width=True, type="primary")
        
    if search_btn and search_query:
        with st.spinner("🔍 Searching..."):
            results = search_engine.rrf_search(query=search_query, k=60, limit=5)
        
        if results:
            st.success(f"Found {len(results)} great matches for you!")
            for idx, result in enumerate(results, 1):
                doc = result['document']
                display_movie_card(idx, doc.get('title', 'Unknown Title'), doc.get('description', 'No description available.'), result.get('rrf_score', 0))
        else:
            st.info("No results found. Try a different query.")

# --- Tab 2: Ask Mode ---
with tab2:
    st.markdown("#### Ask a question and get an AI-generated answer grounded in retrieved movies.")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        ask_query = st.text_input("Ask for a recommendation", placeholder="e.g., What should I watch if I like mind-bending thrillers?", label_visibility="collapsed")
    with col2:
        ask_btn = st.button("Ask AI", use_container_width=True, type="primary")
        
    if ask_btn and ask_query:
        if not client:
            st.error("OpenRouter API key is missing. Please add it to your .env file.")
        else:
            with st.spinner("📚 Retrieving the best movies for context..."):
                results = search_engine.rrf_search(query=ask_query, k=60, limit=5)
            
            if results:
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
                
                with st.spinner("🤖 Generating your tailored recommendation..."):
                    try:
                        response = client.chat.completions.create(
                            model="openrouter/free",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        answer = response.choices[0].message.content
                        
                        st.markdown("### 💡 AI Recommendation")
                        st.info(answer)
                        
                        st.markdown("### 📚 Source Movies")
                        for idx, res in enumerate(results, 1):
                            doc = res['document']
                            display_movie_card(idx, doc.get('title', 'Unknown'), doc.get('description', 'No description.'), res.get('rrf_score', 0))
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
            else:
                st.info("No relevant movies found to answer your question.")
