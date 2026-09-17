from sentence_transformers import SentenceTransformer
import textwrap
from typing import Any, TypedDict

import numpy as np
import re   
from numpy.typing import NDArray
import json
from .search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    MOVIE_EMBEDDINGS_PATH,
    Movie,
    load_movies,
)

model = SentenceTransformer('all-MiniLM-L6-v2')
EmbeddingArray = NDArray[Any]

class SemanticSearch:
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        self.model = SentenceTransformer(model_name)
        self.embedding = None
        self.documents : list[Movie] = []
        self.document_map = {}
        
    def generate_embedding(self, text: str):
        if not text or text.strip() == "":
            raise ValueError("Input text cannot be empty.")
        # list_text = textwrap.wrap(text, width=512)
        return self.model.encode(text)
    
    # list of dictionaries with 'id' and 'text' keys
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        movie_str = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            movie_str.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_str, show_progress_bar=True)
        with open("cache/embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]):
        
        try:
            with open("cache/embeddings.npy", "rb") as f:
                self.embeddings = np.load(f)
            if self.embeddings.shape[0] != len(documents):
                print("Mismatch in number of documents and embeddings. Rebuilding embeddings.")
                self.build_embeddings(documents)
            self.documents = documents
            self.document_map = {doc['id']: doc for doc in documents}
        except FileNotFoundError:
            self.build_embeddings(documents)
        return self.embeddings
    
    def chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
        return semantic_chunk(text, max_chunk_size=chunk_size, overlap=overlap)
    
    def search(self, query: str, limit: int = 5):
        try:
            with open("cache/embeddings.npy", "rb") as f:
                self.embeddings = np.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        
        #list of tuple(similarity, doc)
        similarities = []
        for i, emb in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, emb)
            similarities.append((similarity,self.documents[i]))
        sorted_results = sorted(similarities, key=lambda x: x[0], reverse=True)
        return sorted_results[:limit]

class ChunkedSemanticSearch(SemanticSearch):
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.document_map = {}
    
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunk_list = []
        chunk_metadata = []
        self.document_map = {doc['id']: doc for doc in documents}
        for doc in documents:
            # if doc['description'] is empty, skip it
            if not doc['description'] or doc['description'].strip() == "":
                continue
            chunks = semantic_chunk(doc['description'], max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
            for i in range(len(chunks)):
                chunk_list.append(chunks[i])
                chunk_metadata.append({'movie_idx': doc['id'], 'chunk_idx': i, 'total_chunks': len(chunks)})
        self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True).astype(np.float16)
        self.chunk_metadata = chunk_metadata
        with open("cache/chunk_embeddings.npy", "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata}, f)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        try:
            with open("cache/chunk_embeddings.npy", "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            if len(self.chunk_metadata) != len(self.chunk_embeddings):
                print("Mismatch in number of chunks and embeddings. Rebuilding chunk embeddings.")
                self.build_chunk_embeddings(documents)
            self.documents = documents
            self.document_map = {doc['id']: doc for doc in documents}
        except FileNotFoundError:
            self.build_chunk_embeddings(documents)
            self.documents = documents
            self.document_map = {doc['id']: doc for doc in documents}
        return self.chunk_embeddings

    def search_chunked(self, query: str, limit: int = 10):
        try:
            with open("cache/chunk_embeddings.npy", "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
        except FileNotFoundError:
            raise FileNotFoundError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, emb in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, emb)
            chunk_scores.append({'chunk_idx': i, 'movie_idx': self.chunk_metadata[i]['movie_idx'], 'score': similarity})
        
        movie_best_scores = {}
        for chunk in chunk_scores:
            movie_idx = chunk['movie_idx']
            if movie_idx not in movie_best_scores or chunk['score'] > movie_best_scores[movie_idx]['score']:
                movie_best_scores[movie_idx] = chunk
        
        sorted_results = sorted(movie_best_scores.values(), key=lambda x: x['score'], reverse=True)
        formatted_results = []
        for i in range(min(limit, len(sorted_results))):
            # {
                # "id": doc_id,
                # "title": title,
                # "document": document[:100],
                # "score": round(score, SCORE_PRECISION),
                # "metadata": metadata or {},
            # }
            movie_idx = sorted_results[i]['movie_idx']
            doc = self.document_map[movie_idx]
            formatted_results.append({
                "id": doc['id'],
                "title": doc['title'],
                "document": doc['description'][:100],
                "score": round(sorted_results[i]['score'], 3),
                "metadata": {"chunk_idx": self.chunk_metadata[sorted_results[i]['chunk_idx']]['chunk_idx'], "total_chunks": self.chunk_metadata[sorted_results[i]['chunk_idx']]['total_chunks']}
            })
        return formatted_results
    
def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def verify_embeddings():
    with open("data/movies.json", "r") as f:
        movies = json.load(f)
    semantic_search = SemanticSearch()
    embeddings = semantic_search.load_or_create_embeddings(movies["movies"])
    print(f"Number of docs:   {len(movies["movies"])}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    
def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    formatted_embedding = [f"{(int(x * 1000) / 1000):.3f}" for x in embedding[:3]]
    print(f"First 3 dimensions: {formatted_embedding}")
    print(f"Shape: {embedding.shape}")
    
def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"text: {text}")
    # Formats each number to 3 decimal places
    formatted_embedding = [f"{(int(x * 1000) / 1000):.3f}" for x in embedding[:3]]
    print(f"Embedding for '{text}': {formatted_embedding}")
    print(f"Dimensions: {embedding.shape[0]}")
    

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def semantic_chunk_print(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}: {sentence}")
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks
def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    #remove leading and trailing whitespace
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    #
    fixed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        fixed_sentences.append(sentence)
        
    sentences = fixed_sentences
        
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")