import os
from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    @staticmethod
    def rrf_score(rank: int, k: int = 60) -> float:
        return 1 / (k + rank)

    
    @staticmethod
    def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
        return alpha * bm25_score + (1 - alpha) * semantic_score
    
    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[dict]:
        
        search_limit = 500 * limit  
        
        # * Perform BM25 search
        bm25_results = self._bm25_search(query, search_limit)
        
        # * Perform semantic search
        semantic_results = self.semantic_search.search_chunked(query, search_limit)
        
        
        
        normalized_bm25_scores = self.normalize_scores([score for _, score in bm25_results])
        semantic_scores = [result["score"] for result in semantic_results]
        normalized_semantic_scores = self.normalize_scores( semantic_scores )
        
        # Create a dictionary to hold the combined scores
        combined_scores = {}
        #{ structure of combined_scores:
        #   doc_id: {'document': document, 'bm25_score': bm25_score, 'semantic_score': semantic_score}
        #    
        #}
        for (doc_id, _), normalized_score in zip(
            bm25_results,
            normalized_bm25_scores
        ):
            combined_scores[doc_id] = {
                "document": self.documents[doc_id - 1],
                "bm25_score": normalized_score,
                "semantic_score": 0.0
            }

        for result, normalized_score in zip(semantic_results,normalized_semantic_scores):
            doc_id = result["id"]
        

            if doc_id in combined_scores:
                combined_scores[doc_id]["semantic_score"] = normalized_score
            else:
                combined_scores[doc_id] = {
                    "document": self.documents[doc_id - 1],
                    "bm25_score": 0.0,
                    "semantic_score": normalized_score
                }
        
        results = []
        for doc_id, scores in combined_scores.items():
            hybrid_score = self.hybrid_score(scores["bm25_score"],scores["semantic_score"],alpha)
            results.append({
                "document": scores["document"],
                "bm25_score": scores["bm25_score"],
                "semantic_score": scores["semantic_score"],
                "hybrid_score": hybrid_score
            })
            
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:limit]
        
        
    def rrf_search(self, query: str, k: int = 60, limit: int = 5) -> list[dict]:
        search_limit = 500 * limit

        bm25_results = self._bm25_search(query, search_limit)

        semantic_results = self.semantic_search.search_chunked(query, search_limit)

        combined_results = {}

        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            combined_results[doc_id] = {"doc_id": doc_id,"document": self.documents[doc_id - 1], "bm25_rank": rank, "semantic_rank": None, "rrf_score": self.rrf_score(rank, k)}

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]

            if doc_id in combined_results:
                combined_results[doc_id]["semantic_rank"] = rank
                combined_results[doc_id]["rrf_score"] += self.rrf_score(rank, k)
            else:
                combined_results[doc_id] = {"doc_id": doc_id,"document": self.documents[doc_id - 1], "bm25_rank": None, "semantic_rank": rank, "rrf_score": self.rrf_score(rank, k)}

        results = list(combined_results.values())

        results.sort(key=lambda x: x["rrf_score"], reverse=True)

        return results[:limit]

    def normalize_scores(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]