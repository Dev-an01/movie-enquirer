import argparse
import os
import json
import time
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
model = "openrouter/free"  # or "gpt-4o" if you have access

def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    corrected = (response.choices[0].message.content or "").strip().strip('"')
    return corrected if corrected else query


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.

    User query: "{query}"
    """

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    rewritten = (response.choices[0].message.content or "").strip().strip('"')
    return rewritten if rewritten else query


def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    Output only the additional terms; they will be appended to the original query.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    User query: "{query}"
    """

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    expanded_terms = (response.choices[0].message.content or "").strip().strip('"')
    return f"{query} {expanded_terms}".strip()

def rerank_query(query: str, results: list[dict]) -> list[dict]:
    reranked_results = []

    for result in results:
        doc = result["document"]

        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )

                score_text = (response.choices[0].message.content or "").strip()
                score = float(score_text)

                if 0 <= score <= 10:
                    break

            except Exception:
                pass

        result["rerank_score"] = score
        reranked_results.append(result)

        time.sleep(3)

    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked_results

def batch_rerank_query(query: str, results: list[dict]) -> list[dict]:
    doc_list = []

    for result in results:
        doc = result["document"]
        doc_list.append(
            f"ID: {doc.get('id')} - {doc.get('title', '')} - {doc.get('description', '')}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

            ranked_ids = json.loads(
                (response.choices[0].message.content or "").strip()
            )

            if isinstance(ranked_ids, list):
                break

        except Exception:
            pass

    rank_map = {
        movie_id: rank
        for rank, movie_id in enumerate(ranked_ids, start=1)
    }

    for result in results:
        movie_id = result["document"].get("id")
        result["rerank_rank"] = rank_map.get(movie_id, len(results) + 1)

    results.sort(key=lambda x: x["rerank_rank"])

    return results

def enhance_query(
    query: str, method: Literal["spell", "rewrite", "expand"] | None = None
) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query