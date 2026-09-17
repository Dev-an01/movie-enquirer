import argparse
import json 
import os
from dotenv import load_dotenv
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from openai import OpenAI
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
model = "openrouter/free" 

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    summary_parser = subparsers.add_parser("summarize", help="Summarize a given text")
    citation_parser = subparsers.add_parser("citations", help="Generate citations for a given text")
    question_parser = subparsers.add_parser("question", help="Answer a question based on retrieved documents")
    
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summary_parser.add_argument("query", type=str, help="Text to summarize")
    summary_parser.add_argument("--limit", type=int, default=5, help="Number of top documents to consider for summarization")
    citation_parser.add_argument("query", type=str, help="Text to generate citations for")
    citation_parser.add_argument("--limit", type=int, default=5, help="Number of top documents to consider for citation generation")
    question_parser.add_argument("question", type=str, help="Question to answer based on retrieved documents")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of top documents to consider for answering the question")
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            documents = load_movies()

            hybrid_search = HybridSearch(documents)

            results = hybrid_search.rrf_search(
                query=query,
                k=60,
                limit=5,
            )

            docs = "\n".join(
                f"- {result['document']['title']}: {result['document']['description']}"
                for result in results
            )

            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
        Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
        Provide a comprehensive answer that addresses the user's query.

        Query: {query}

        Documents:
        {docs}

        Answer:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )   

            answer = response.choices[0].message.content

            print("Search Results:")
            for result in results:
                print(f"- {result['document']['title']}")

            print()

            print("RAG Response:")
            print(answer)
            
        case "summarize":
            query = args.query
            limit = args.limit
            
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query=query,k=60,limit=limit)
            
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content
            
            print("Search Results:")
            for result in results:
                print(f"- {result['document']['title']}")
                
            print("LLM Summary:")
            print(answer)
        case "citations":
            query = args.query
            limit = args.limit
            
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query=query,k=60,limit=limit)
            
            docs = "\n".join(
                f"[{i}] {result['document']['title']}: {result['document']['description']}"
                for i, result in enumerate(results, start=1)
            )
            
            prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content
            
            print("Search Results:")
            for result in results:
                print(f"- {result['document']['title']}")
                
            print("LLM Answer:")
            print(answer)
        
        case "question":
            question = args.question
            limit = args.limit
            
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query=question,k=60,limit=limit)
            
            docs = "\n".join(
                f"- {result['document']['title']}: {result['document']['description']}"
                for result in results
            )
            prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{docs}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content
            
            print("Search Results:")
            for result in results:
                print(f"- {result['document']['title']}")
                
            print("Answer:")
            print(answer)
        case _:
            parser.print_help()
            

if __name__ == "__main__":
    main()