# Movie RAG Search Engine

A learning project that builds a movie search and Retrieval-Augmented Generation (RAG) pipeline from first principles.

The project searches 5,000 movie records with keyword search, BM25, dense embeddings, semantic chunking, and hybrid ranking. It can then pass the retrieved movies to an LLM through OpenRouter to produce grounded recommendations, summaries, citations, and answers.

> This repository is currently a command-line learning project. A simple demo UI and hosting path are planned, but not implemented yet.

## What this project demonstrates

- Text preprocessing with punctuation removal, stop-word filtering, and Porter stemming
- An inverted index with term frequency, inverse document frequency, TF-IDF, and BM25
- Dense semantic search with `sentence-transformers/all-MiniLM-L6-v2`
- Description chunking and per-movie score aggregation
- Hybrid retrieval with weighted score fusion and Reciprocal Rank Fusion (RRF)
- Query spell correction, rewriting, and expansion through an LLM
- LLM-based and cross-encoder reranking
- Retrieval-augmented movie recommendations and question answering
- Image-to-movie search with CLIP
- Precision, recall, and F1 evaluation against a small golden dataset

## How it works

```text
User query
   |
   +--> Text preprocessing --> BM25 search ---------+
   |                                                |
   +--> MiniLM embedding --> chunk similarity ------+--> RRF or weighted fusion
                                                        |
                                                        +--> optional reranking
                                                        |
                                                        +--> top movie documents
                                                                  |
                                                                  +--> OpenRouter LLM
                                                                        |
                                                                        +--> grounded answer
```

The default RAG path uses RRF to merge two ranked lists:

1. BM25 rewards exact and rare term matches.
2. Semantic search rewards descriptions whose meaning is close to the query.
3. RRF combines ranks instead of trying to compare two unrelated score scales.
4. The five highest-ranked movies become context for the LLM.

## Technology choices

| Tool | Used for | Why it is here |
|---|---|---|
| Python 3.14 | Application language | Clear data and ML ecosystem, plus fast iteration for a learning project |
| `uv` | Environment and dependency management | Reproducible installs from `pyproject.toml` and `uv.lock` |
| NLTK | Porter stemming | Reduces related word forms to a shared stem for lexical retrieval |
| NumPy | Embedding storage and cosine similarity | Simple local vector math without a vector database |
| Sentence Transformers | Text and image embedding models | Provides MiniLM for text and CLIP for multimodal retrieval |
| OpenAI Python SDK | OpenRouter client | OpenRouter exposes an OpenAI-compatible chat-completions API |
| OpenRouter | Hosted LLM access | Supports query enhancement, reranking, evaluation, and answer generation without running an LLM locally |
| Pillow | Image loading | Opens images before CLIP creates an image embedding |
| Pickle and `.npy` files | Local indexes and embedding caches | Keeps the learning version small and avoids operating a database |

## Project structure

```text
rag-search-engine/
├── cli/
│   ├── augmented_generation_cli.py  # RAG, summaries, citations, Q&A
│   ├── describe_image_cli.py         # image + text query rewriting
│   ├── evaluation_cli.py             # precision, recall, F1
│   ├── hybrid_search_cli.py          # weighted fusion, RRF, reranking
│   ├── keyword_search_cli.py         # inverted-index and BM25 commands
│   ├── multimodal_search_cli.py      # CLIP image search
│   ├── query_enhancement.py          # LLM rewrite and rerank helpers
│   ├── semantic_search_cli.py        # embeddings and semantic search
│   └── lib/
│       ├── hybrid_search.py
│       ├── inverted_index.py
│       ├── keyword_search.py
│       ├── multimodal_search.py
│       ├── search_utils.py
│       ├── semantic_search.py
│       └── text_utils.py
├── data/                              # local movie and evaluation data
├── cache/                             # generated indexes and embeddings
├── blog.md                            # project walkthrough and lessons
├── pyproject.toml
└── uv.lock
```

## Setup

### Prerequisites

- Python 3.14
- [`uv`](https://docs.astral.sh/uv/)
- An OpenRouter API key for LLM-backed commands
- Internet access on the first run so Sentence Transformers can download model weights

### 1. Install dependencies

From the project root:

```bash
uv sync
```

The locked environment includes NLTK, NumPy, Pillow, python-dotenv, the OpenAI SDK, and Sentence Transformers.

### 2. Add the movie data

The `data/` directory is intentionally ignored by Git. The application expects these local files:

```text
data/
├── movies.json
├── stopwords.txt
├── golden_dataset.json
└── paddington.jpeg          # optional example image
```

`movies.json` must use this shape:

```json
{
  "movies": [
    {
      "id": 1,
      "title": "Example Movie",
      "description": "A short plot description used for search."
    }
  ]
}
```

Use consecutive integer IDs starting at `1`. Parts of the current implementation map a movie ID back to `documents[id - 1]`.

`stopwords.txt` contains one word per line. `golden_dataset.json` is optional unless you run the evaluation command:

```json
{
  "test_cases": [
    {
      "query": "a survival story in space",
      "relevant_docs": ["The Martian", "Interstellar"]
    }
  ]
}
```

### 3. Configure OpenRouter

Create a local `.env` file. It is ignored by Git.

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Only augmented generation, query enhancement, LLM reranking, LLM evaluation, and image description logically need this key. Keyword, BM25, semantic, hybrid retrieval, cross-encoder reranking, and CLIP image search run locally. One current CLI quirk is that `hybrid_search_cli.py` checks for `OPENROUTER_API_KEY` at startup even when you select a local-only hybrid command, so export the key before using that CLI until the check is moved into the LLM-only branches.

### 4. Build the local search artifacts

Build the inverted index:

```bash
uv run python cli/keyword_search_cli.py build
```

Build whole-movie embeddings:

```bash
uv run python cli/semantic_search_cli.py verify_embeddings
```

Build chunk embeddings used by hybrid search:

```bash
uv run python cli/semantic_search_cli.py embed_chunks
```

Generated files are stored in `cache/`. Rebuild them whenever the movie dataset or chunking behavior changes.

### 5. Run a first search

```bash
uv run python cli/keyword_search_cli.py bm25search "space adventure"
```

Example output from the current dataset:

```text
Searching for: space adventure
1. Space Ace (Score: 10.65)
2. The Adventures of the Galaxy Rangers (Score: 9.53)
3. Titanfall (Score: 9.12)
```

## Common commands

### Keyword and BM25 search

```bash
# Build or rebuild the inverted index
uv run python cli/keyword_search_cli.py build

# Simple token-based lookup
uv run python cli/keyword_search_cli.py search "space adventure"

# Ranked lexical search
uv run python cli/keyword_search_cli.py bm25search "space adventure"

# Inspect index statistics
uv run python cli/keyword_search_cli.py tf 1 "space"
uv run python cli/keyword_search_cli.py idf "space"
uv run python cli/keyword_search_cli.py tfidf 1 "space"
uv run python cli/keyword_search_cli.py bm25idf "space"
uv run python cli/keyword_search_cli.py bm25tf 1 "space"
```

### Semantic search

```bash
# Inspect the MiniLM model
uv run python cli/semantic_search_cli.py verify

# Embed text or a query
uv run python cli/semantic_search_cli.py embed_text "a lonely astronaut"
uv run python cli/semantic_search_cli.py embed_query "a lonely astronaut"

# Whole-document semantic search
uv run python cli/semantic_search_cli.py search "a lonely astronaut" --limit 5

# Chunked semantic search
uv run python cli/semantic_search_cli.py search_chunked "a lonely astronaut" --limit 5
```

### Hybrid search

The underlying hybrid retriever is local, but the current CLI performs an eager OpenRouter-key check. Export the key before running these commands:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

```bash
# Weighted BM25 + semantic scores
uv run python cli/hybrid_search_cli.py weighted-search "dream inside a dream" --alpha 0.5 --limit 5

# Reciprocal Rank Fusion
uv run python cli/hybrid_search_cli.py rrf-search "dream inside a dream" --k 60 --limit 5

# Optional LLM query enhancement
uv run python cli/hybrid_search_cli.py rrf-search "scarry bear film" --enhance spell
uv run python cli/hybrid_search_cli.py rrf-search "that bear movie with leo" --enhance rewrite
uv run python cli/hybrid_search_cli.py rrf-search "funny bear" --enhance expand

# Optional reranking
uv run python cli/hybrid_search_cli.py rrf-search "space survival" --rerank-method batch
uv run python cli/hybrid_search_cli.py rrf-search "space survival" --rerank-method cross_encoder
```

The `individual` and `batch` rerankers call OpenRouter. The `cross_encoder` reranker downloads and runs `cross-encoder/ms-marco-TinyBERT-L2-v2` locally.

### Retrieval-Augmented Generation

```bash
uv run python cli/augmented_generation_cli.py rag "What should I watch if I like mind-bending thrillers?"
uv run python cli/augmented_generation_cli.py summarize "funny animated adventures" --limit 5
uv run python cli/augmented_generation_cli.py citations "movies about surviving alone" --limit 5
uv run python cli/augmented_generation_cli.py question "Which retrieved movie is the darkest?" --limit 5
```

These commands retrieve movies first, place their titles and descriptions into the prompt, and then ask the LLM to answer from that context.

### Multimodal search

```bash
# Search directly with a CLIP image embedding
uv run python cli/multimodal_search_cli.py image_search data/paddington.jpeg

# Use an image and text to rewrite a movie query through a vision-capable LLM
uv run python cli/describe_image_cli.py --image data/paddington.jpeg --query "find movies like this"
```

### Evaluation

```bash
uv run python cli/evaluation_cli.py --limit 5
```

The evaluator compares retrieved movie titles with `relevant_docs` in `data/golden_dataset.json` and prints precision@k, recall@k, and F1 for each test case.

## Suggested learning path

If you are using the repository to learn RAG, follow the code in this order:

1. `cli/lib/text_utils.py`: normalize and tokenize text.
2. `cli/lib/inverted_index.py`: build exact-term retrieval and BM25.
3. `cli/lib/semantic_search.py`: turn text into dense vectors and compare them with cosine similarity.
4. `cli/lib/hybrid_search.py`: combine lexical and semantic rankings.
5. `cli/query_enhancement.py`: improve queries and rerank candidates.
6. `cli/augmented_generation_cli.py`: add the generation step that turns retrieval into RAG.
7. `cli/evaluation_cli.py`: measure whether retrieval changes help.

For the longer explanation, read [blog.md](blog.md).

## Demo UI and hosting plan

This is a plan only. No UI or deployment files have been added yet.

### Recommended first version

Build one Streamlit app that imports the existing `HybridSearch` class. Keep the search engine in the same process. Do not add FastAPI, a separate frontend, a vector database, authentication, or chat history for the first public demo.

The page should have:

- A clear title and one-sentence explanation
- A large natural-language search box
- Example-query chips such as “space survival,” “funny bear movie,” and “mind-bending thriller”
- A Search mode that works without an API key and shows ranked movie cards
- An Ask mode that uses OpenRouter and shows the generated answer above its retrieved sources
- Optional controls in a collapsed “How retrieval works” panel: result count, RRF versus weighted fusion, query enhancement, and reranking
- Each movie card showing title, short description, lexical rank, semantic rank, and final RRF score
- Friendly loading, empty, and error states

### Implementation phases

1. **Make retrieval app-safe**
   - Move model and index initialization behind cached functions.
   - Replace working-directory-relative paths with the existing project-root constants.
   - Return structured results and errors instead of printing inside library code.
   - Add a smoke test for one BM25, semantic, hybrid, and RAG query.

2. **Add `app.py` with Streamlit**
   - Reuse `HybridSearch`; do not duplicate retrieval logic.
   - Cache the dataset, MiniLM model, embeddings, and index with Streamlit resource caching.
   - Keep OpenRouter optional so the retrieval demo still works when the API is unavailable.

3. **Prepare demo assets**
   - Commit a small redistributable demo dataset or add a deterministic build-time download. The current `data/` directory is ignored by Git, so a fresh deployment will not contain it.
   - Either commit the generated caches or rebuild them during deployment. Committing a reduced demo cache gives a faster and more predictable cold start.
   - Add `.env.example`; keep the real key out of Git.

4. **Deploy to Streamlit Community Cloud first**
   - Push the repository to GitHub and select `app.py` as the entry point.
   - Store `OPENROUTER_API_KEY` in Streamlit’s app secrets, not in the repository.
   - Pin a supported Python version and verify that the MiniLM model fits the available memory.
   - Expect cold starts and treat this as a portfolio/demo deployment, not an always-on production service.

5. **Verify the public demo**
   - Test desktop and mobile layouts.
   - Test a lexical query, a conceptual query, a misspelled query, and an out-of-domain query.
   - Confirm every generated answer displays its retrieved movie sources.
   - Confirm the app still offers Search mode if OpenRouter is down or rate-limited.

Streamlit Community Cloud deploys from GitHub and supports app-level secret management. Its current environment and deployment limits are documented in the [Community Cloud status guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/status) and [secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

### When to choose another host

Use a Hugging Face Docker Space if you need full container control or want the demo next to model assets. As of September 2026, Hugging Face documents Gradio and Docker Spaces as compute-backed products that require an eligible paid plan, with limited ZeroGPU exceptions. Docker Spaces expose one public application port, inject runtime secrets as environment variables, and use non-persistent disk unless you add storage. See the official [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) and [Docker Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-docker).

Split the app into FastAPI plus a separate frontend only when another client needs a stable API or independent scaling. That architecture is unnecessary for the first demo.

## Current limitations

- The project uses local files rather than a vector database.
- Index construction assumes movie IDs are consecutive and aligned with list positions.
- Several cache paths are relative to the current working directory, so commands should be run from the project root.
- The text model is created at module import time and again inside search classes, which can slow startup.
- LLM calls use the moving `openrouter/free` model route, so output quality and model availability can vary.
- LLM reranking retries indefinitely on malformed responses and should receive bounded retries before hosting.
- Generated answers are prompted to use retrieved documents, but citations are not programmatically validated.
- The evaluation set has only 10 test cases and measures title-level retrieval relevance, not answer faithfulness.

## Troubleshooting

### `OPENROUTER_API_KEY environment variable not set`

Add the key to `.env` for CLIs that call `load_dotenv()`. `hybrid_search_cli.py` currently reads the process environment directly, so export the key before using its enhancement, LLM reranking, or LLM evaluation options:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

### `data/movies.json` was not found

Run commands from the repository root and confirm the dataset uses the structure shown in the setup section.

### `Index or docmap file not found`

Build the lexical index:

```bash
uv run python cli/keyword_search_cli.py build
```

### Embedding count does not match the dataset

Delete or replace the derived embedding files and run the embedding build commands again. Never reuse caches created for a different `movies.json`.

### The first semantic command is slow

Sentence Transformers downloads model weights on first use. Later runs use local model and embedding caches.

## Further reading

- [Project walkthrough and lessons](blog.md)
- [Sentence Transformers documentation](https://www.sbert.net/)
- [OpenRouter API documentation](https://openrouter.ai/docs)

## License

No license file is included yet. Add one before inviting others to reuse or redistribute the code or dataset.
