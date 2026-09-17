# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

The existing RAG system is Python 3.14 with local lexical and semantic retrieval, Sentence Transformers, and optional OpenRouter generation. The presentation layer will use Streamlit and target Streamlit Community Cloud for the first hosted demo.

## Users

The primary audience is recruiters and developers inspecting the author's work. They arrive from a resume, portfolio, GitHub profile, or direct project link and need to judge quickly whether the project works and whether the author understands its retrieval pipeline.

## Product Purpose

The project teaches and demonstrates a movie Retrieval-Augmented Generation system built from first principles. The public demo must let a visitor run a natural-language query, see a useful answer, and inspect the lexical, semantic, fusion, and source evidence behind it.

Success means a recruiter can understand the project and see it work in under two minutes, while a developer can inspect enough retrieval detail to evaluate the implementation rather than treating it as an opaque chat wrapper.

## Positioning

This is an inspectable learning implementation, not a framework-generated chatbot or a movie-streaming product. It exposes the mechanics the author implemented: preprocessing, an inverted index, BM25, dense embeddings, semantic chunking, hybrid fusion, reranking experiments, evaluation, and grounded answer generation.

## Operating Context

Visitors use a hosted Streamlit demo and may continue to the GitHub repository and technical article. The core demonstration is a natural-language movie query followed by ranked evidence and an optional generated answer. The presentation must also support a reliable retrieval-only path when OpenRouter is unavailable.

## Capabilities and Constraints

- The current corpus contains 5,000 movie records with IDs, titles, and descriptions.
- BM25 and chunked MiniLM retrieval are combined with weighted fusion or Reciprocal Rank Fusion.
- OpenRouter is optional and is used for generated answers and selected enhancement or reranking experiments.
- The first demo remains a single Streamlit process. It does not add FastAPI, a separate frontend, authentication, user accounts, or conversational memory.
- The existing RAG implementation was written line by line by the author, with ordinary editor autocomplete assistance.
- AI assistance is authorized for the new Streamlit presentation layer and its integration work. Public wording must preserve this boundary accurately.
- The UI must not claim benchmarks, production readiness, dataset rights, or capabilities that cannot be demonstrated from the repository.

## Evidence on Hand

- Working BM25 command and local cache artifacts.
- Search and RAG CLI entry points under `cli/`.
- A 5,000-record local movie dataset and 10-case golden evaluation set under `data/`.
- Architecture, setup, limitations, and hosting notes in `README.md`.
- A technical walkthrough in `blog.md`.
- No existing graphical interface, screenshots, testimonials, production usage, or public deployment exists yet.

## Product Principles

1. Show the mechanism before making claims about it.
2. Keep generated answers visibly connected to retrieved movie sources.
3. Give recruiters a fast success path and developers an optional inspection path.
4. Preserve the hand-built RAG as the artifact; treat the AI-assisted UI as presentation infrastructure.
5. Prefer one reliable demo flow over a wide but fragile feature tour.

## Accessibility & Inclusion

The demo must be keyboard usable, maintain readable contrast, avoid color-only relevance indicators, expose meaningful loading and error text, and remain usable on laptop and mobile widths.

## Showcase Plan

The primary demonstration vehicle is a **plain Streamlit showcase** (the preferred plan). This UI will provide:
1. **Search Mode**: A simple search bar querying the local lexical and semantic index.
2. **Ask Mode**: A conversational UI generating answers from OpenRouter while showing retrieved sources.
3. **Reference Links**: Direct access to the code and the technical walkthrough in [blog.md](blog.md).
