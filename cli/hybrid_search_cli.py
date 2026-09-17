import argparse
import os,json
from dotenv import load_dotenv
from openai import OpenAI
from query_enhancement import enhance_query,rerank_query,batch_rerank_query
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("verify", help="Verify the semantic search model")
    
    normalize_parser = subparsers.add_parser("normalize", help="Perform normalised hybrid search")
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search")
    
    
    normalize_parser.add_argument("list_of_scores", nargs= "+", type = float, help="List of scores to normalise")
    weighted_search_parser.add_argument("query", type=str, help="Query to search for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 score (default: 0.5)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return (default: 5)")
    rrf_search_parser.add_argument("query", type=str, help="Query to search for")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="RRF parameter k (default: 60)")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return (default: 5)")
    rrf_search_parser.add_argument("--enhance", type = str, choices = ["spell","rewrite","expand"], help = "Enhance the query using spell correction")
    rrf_search_parser.add_argument("--rerank-method", type = str, choices = ["individual","batch","cross_encoder"], help = "Reranking method to use for final results")
    rrf_search_parser.add_argument("--evaluate", action = "store_true", help = "Evaluate the results using precision, recall and F1 score")
    args = parser.parse_args()

    match args.command:
        case "normalize":
            list_of_scores = args.list_of_scores
            if not list_of_scores:
                print("Please provide a list of scores to normalise.")
                return
            # Normalise the scores
            min_score = min(list_of_scores)
            max_score = max(list_of_scores)
            if min_score == max_score:
                normalised_scores = [1.0 for _ in list_of_scores]
            else:
                normalised_scores = [(score - min_score) / (max_score - min_score) for score in list_of_scores]
            
            for score in normalised_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            
            documents = load_movies()

            hybrid_search = HybridSearch(documents)

            results = hybrid_search.weighted_search(query=args.query, alpha=args.alpha, limit=args.limit)

            for i, result in enumerate(results, start=1):
                document = result["document"]

                print(f"{i}. {document['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"  BM25: {result['bm25_score']:.3f}, "
                    f"Semantic: {result['semantic_score']:.3f}"
                )
                print(
                    f"  {document['description'][:200]}..."
                )
        
        case "rrf-search":
            documents = load_movies()
            query = args.query

            #debug
            logger.info(f"Original query: {query}")
            
            if args.enhance:
                enhanced_query = enhance_query(query, args.enhance)
                logger.info(f"Enhanced query: {enhanced_query}")
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query

            hybrid_search = HybridSearch(documents)

            search_limit = 5*args.limit if args.rerank_method else args.limit
            results = hybrid_search.rrf_search(query=query, k=args.k, limit=search_limit)
            logger.info(
                f"Results after RRF search: "
                f"{[r['document']['title'] for r in results]}"
            )
            if args.rerank_method == "individual":
                print(f"Re-ranking top {args.limit} results using individual method...")
                results = rerank_query(query, results)
                results = results[:args.limit]
            elif args.rerank_method == "batch":
                print(f"Re-ranking top {args.limit} results using batch method...")
                results = batch_rerank_query(query, results)
                results = results[:args.limit]
            elif args.rerank_method == "cross_encoder":
                pairs = []
                for result in results:
                    pairs.append((query, result["document"]["title"] + " " + result["document"]["description"]))
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                score = cross_encoder.predict(pairs)
                sorted_results = sorted(zip(results, score), key=lambda x: x[1], reverse=True)
                results = [result for result, _ in sorted_results][:args.limit]
                
            logger.info(
                f"Final results after re-ranking: "
                f"{[r['document']['title'] for r in results]}"
            )   

            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
            print()

            for i, result in enumerate(results, start=1):
                document = result["document"]

                print(f"{i}. {document['title']}")

                if args.rerank_method == "individual":
                    print(f"   Re-rank Score: {result['rerank_score']:.3f}/10")
                elif args.rerank_method == "batch":
                    print(f"   Re-rank Rank: {result['rerank_rank']}")

                print(f"   RRF Score: {result['rrf_score']:.3f}")
                print(
                    f"   BM25 Rank: {result['bm25_rank']}, "
                    f"Semantic Rank: {result['semantic_rank']}"
                )
                print(f"   {document['description'][:200]}...")
                print()
            if args.evaluate:
                    formatted_results = [
                        f"{i}. {result['document']['title']}: "
                        f"{result['document']['description']}"
                        for i, result in enumerate(results, start=1)
                    ]

                    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""

                    response = client.chat.completions.create(
                        model="openrouter/free",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    )

                    scores = json.loads(
                        response.choices[0].message.content
                    )

                    print("Evaluation:")

                    for i, (result, score) in enumerate(
                        zip(results, scores),
                        start=1,
                    ):
                        print(
                            f"{i}. "
                            f"{result['document']['title']}: "
                            f"{score}/3"
                        )

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()