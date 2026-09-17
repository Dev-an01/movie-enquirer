
from .search_utils import DEFAULT_SEARCH_LIMIT
from .inverted_index import InvertedIndex
from .text_utils import tokenize_text

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    
    index = InvertedIndex()
    try: 
        index.load()
    except:
        return []
    
    query_tokens = tokenize_text(query)
    doc_ids = set()
    for token in query_tokens:
        if token in index.index:
            doc_ids.update(index.get_document(token))
    # for i in doc_ids:
    #     print(f"doc_id: {i+1}, title: {index.docmap[i-1]}")
    results = []
    for doc_id in sorted(doc_ids):
        movie = index.movies[doc_id-1]
        if movie:
            results.append(movie)
        if len(results) >= limit:
                break
    return results

    # search when we have not saved the index yet
    # movies = load_movies()
    # results = []
    # for movie in movies:
    #     query_tokens = tokenize_text(query)
    #     title_tokens = tokenize_text(movie["title"])
    #     if has_matching_token(query_tokens, title_tokens):
    #         results.append(movie)
    #         if len(results) >= limit:
    #             break
        
    # return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

