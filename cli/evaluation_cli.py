import argparse
import json
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit",type=int,default=5,help="Number of results to evaluate (k for precision@k, recall@k)",)

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open("data/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    
    print(f"k={limit}")
    print()

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant = test_case["relevant_docs"]

        results = hybrid_search.rrf_search(
            query=query,
            k=60,
            limit=args.limit,
        )

        retrieved = [result["document"]["title"] for result in results]

        if not relevant:
            precision = 0.0
        else:
            relevant_set = set(relevant)
            retrieved_set = set(retrieved)
            true_positives = len(relevant_set.intersection(retrieved_set))
            precision = true_positives / len(retrieved) if retrieved else 0.0
            recall = true_positives / len(relevant) if relevant else 0.0
            f1_score = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{args.limit}: {precision:.4f}")
        print(f"  - Recall@{args.limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1_score:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant)}")
        print()

if __name__ == "__main__":
    main()