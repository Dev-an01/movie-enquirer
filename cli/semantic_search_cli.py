import argparse
from lib.semantic_search import verify_model
from lib.semantic_search import embed_text , verify_embeddings, embed_query_text, SemanticSearch, ChunkedSemanticSearch
import json
import re


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")   
    
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("verify", help="Verify the semantic search model")
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    verify_embed_parser = subparsers.add_parser("verify_embeddings", help="Verify the embedding generation process")
    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding for a given query")
    search_parser = subparsers.add_parser("search", help="Search for documents based on a query")
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text into smaller pieces")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a given text into smaller pieces based on semantic similarity")
    embed_chunk_parser = subparsers.add_parser("embed_chunks", help="Generate embeddings for a given chunk of text")
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for documents based on a query using chunked embeddings")
    # Add arguments
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")
    embed_query_parser.add_argument("text", type=str, help="Query to generate embedding for")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return (default: 5)")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk (default: 100)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap size between chunks (default: 50)")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=200, help="Size of each chunk (default: 100)")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap size between chunks (default: 50)")
    search_chunked_parser.add_argument("query", type=str, help="Query to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return (default: 5)")
    args = parser.parse_args()
    
    match args.command:
        case "verify":
            verify_model() # 4. Call the correctly named function
        case "embed_text":
            embed_text(args.text) # 5. Call the correctly named function
        case "verify_embeddings":
            verify_embeddings() # 6. Call the correctly named function
        case "embed_query":
            embed_query_text(args.text) # 7. Call the correctly named function
        case "search":
            semantic_search = SemanticSearch()
            with open("data/movies.json", "r") as f:
                movies_data = json.load(f)
            semantic_search.load_or_create_embeddings(movies_data["movies"])
            
            results = semantic_search.search(args.query, limit=args.limit)
            for similarity, doc in results:
                print(f"{doc['title']} (score: {similarity:.3f}) {doc['description']}")
                
        case "chunk":
            chunks = args.text.split()
            print(f"Chunking {len(args.text)} characters")
            cnt = 1
            for i in range(0, len(chunks), args.chunk_size - args.overlap):
                chunk = ' '.join(chunks[i:i + args.chunk_size])
                print(f"{cnt}. {chunk}")
                cnt += 1
                
        case "semantic_chunk":
            semantic_search = SemanticSearch()
            chunks = semantic_search.chunk_text(args.text)
            # print(f"Chunking {len(args.text)} characters into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        case "embed_chunks":
            chunked_semantic_search = ChunkedSemanticSearch()
            with open("data/movies.json", "r") as f:
                movies_data = json.load(f)
            embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies_data["movies"])
            print(f"Generated {len(embeddings)} chunked embeddings")
            
        case "search_chunked":
            chunked_semantic_search = ChunkedSemanticSearch()
            with open("data/movies.json", "r") as f:
                movies_data = json.load(f)
            chunked_semantic_search.load_or_create_chunk_embeddings(movies_data["movies"])
            results = chunked_semantic_search.search_chunked(args.query, limit=args.limit)
#             print(f"\n{i}. {TITLE} (score: {SCORE:.4f})")
#             print(f"   {DOCUMENT}...")
            for i in range(len(results)):
                # {
                    # "id": doc_id,
                    # "title": title,
                    # "document": document[:100],
                    # "score": round(score, SCORE_PRECISION),
                    # "metadata": metadata or {},
                # }
                similarity,doc = results[i]['score'], results[i]
                TITLE = doc['title']
                DOCUMENT = doc['document']
                SCORE = similarity
                print(f"\n{i+1}. {TITLE} (score: {SCORE:.4f})")
                print(f"   {DOCUMENT}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()