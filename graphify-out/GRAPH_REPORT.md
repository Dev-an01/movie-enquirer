# Graph Report - rag-search-engine  (2026-09-17)

## Corpus Check
- 19 files · ~443,383 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: .pkl 4, (none) 3, .npy 2)

## Summary
- 126 nodes · 295 edges · 10 communities (7 shown, 3 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Semantic Search and Chunking
- LLM Query Processing
- Multimodal Search and Evaluation
- BM25 Inverted Index
- Shared Data Types
- Keyword Search CLI
- Hybrid Search Fusion
- Index Scoring Commands
- Text Preprocessing
- Project Root

## God Nodes (most connected - your core abstractions)
1. `InvertedIndex` - 28 edges
2. `HybridSearch` - 17 edges
3. `load_movies()` - 16 edges
4. `SemanticSearch` - 15 edges
5. `tokenize_text()` - 12 edges
6. `ChunkedSemanticSearch` - 11 edges
7. `tokenize_single_term()` - 8 edges
8. `MultiModalSearch` - 7 edges
9. `main()` - 7 edges
10. `main()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `HybridSearch` --uses--> `InvertedIndex`  [INFERRED]
  cli/lib/hybrid_search.py → cli/lib/inverted_index.py
- `HybridSearch` --uses--> `ChunkedSemanticSearch`  [INFERRED]
  cli/lib/hybrid_search.py → cli/lib/semantic_search.py
- `main()` --calls--> `HybridSearch`  [EXTRACTED]
  cli/augmented_generation_cli.py → cli/lib/hybrid_search.py
- `main()` --calls--> `load_movies()`  [EXTRACTED]
  cli/augmented_generation_cli.py → cli/lib/search_utils.py
- `main()` --calls--> `HybridSearch`  [EXTRACTED]
  cli/evaluation_cli.py → cli/lib/hybrid_search.py

## Import Cycles
- None detected.

## Communities (10 total, 3 thin omitted)

### Community 0 - "Semantic Search and Chunking"
Cohesion: 0.13
Nodes (18): Movie, chunk_text(), ChunkedSemanticSearch, cosine_similarity(), embed_query_text(), embed_text(), fixed_size_chunking(), semantic_chunk() (+10 more)

### Community 1 - "LLM Query Processing"
Cohesion: 0.18
Nodes (16): argparse, base64, main(), main(), batch_rerank_query(), enhance_query(), expand_query(), rerank_query() (+8 more)

### Community 2 - "Multimodal Search and Evaluation"
Cohesion: 0.21
Nodes (8): main(), MultiModalSearch, search_image_command(), load_movies(), main(), json, pil, sentence_transformers

### Community 4 - "Shared Data Types"
Cohesion: 0.29
Nodes (9): Any, format_search_result(), GoldenDataset, GoldenTestCase, load_golden_dataset(), Create standardized search result Args: doc_id: Document ID title: Document…, SearchResult, TypedDict (+1 more)

### Community 5 - "Keyword Search CLI"
Cohesion: 0.39
Nodes (5): main(), build_command(), search_command(), tf_tokenize(), math

### Community 7 - "Index Scoring Commands"
Cohesion: 0.43
Nodes (7): bm25_idf_command(), bm25_tf_command(), idf_command(), tf_command(), tfidf_command(), tokenize_single_term(), pickle

### Community 8 - "Text Preprocessing"
Cohesion: 0.39
Nodes (7): load_stopwords(), preprocess_text(), remove_stopwords(), stem_tokens(), tokenize_text(), nltk_stem, string

## Knowledge Gaps
- **1 isolated node(s):** `rag-search-engine`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 24 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `InvertedIndex` connect `BM25 Inverted Index` to `Multimodal Search and Evaluation`, `Keyword Search CLI`, `Hybrid Search Fusion`, `Index Scoring Commands`?**
  _High betweenness centrality (0.227) - this node is a cross-community bridge._
- **Why does `load_movies()` connect `Multimodal Search and Evaluation` to `Semantic Search and Chunking`, `LLM Query Processing`, `BM25 Inverted Index`, `Shared Data Types`, `Index Scoring Commands`, `Text Preprocessing`?**
  _High betweenness centrality (0.146) - this node is a cross-community bridge._
- **Why does `HybridSearch` connect `Hybrid Search Fusion` to `Semantic Search and Chunking`, `LLM Query Processing`, `Multimodal Search and Evaluation`, `BM25 Inverted Index`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `HybridSearch` (e.g. with `InvertedIndex` and `ChunkedSemanticSearch`) actually correct?**
  _`HybridSearch` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `rag-search-engine` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Semantic Search and Chunking` be split into smaller, more focused modules?**
  _Cohesion score 0.13118279569892474 - nodes in this community are weakly interconnected._