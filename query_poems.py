import os
import json
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# ------------ CONFIG & SETUP ------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sar-poetry")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing from .env")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

EMBED_MODEL = "text-embedding-3-large"


def embed_text(text: str) -> List[float]:
    """Get an embedding vector for a query string."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return response.data[0].embedding


def search_poems(query: str, top_k: int = 5) -> List[Dict]:
    """Semantic search in Pinecone for the closest poems to the query."""
    vector = embed_text(query)

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    poems = []
    for match in results.matches:
        meta = match.metadata or {}
        poems.append({
            "id": match.id,
            "score": match.score,
            "title": meta.get("title"),
            "author": meta.get("author"),
            "url": meta.get("url"),
            "moods": meta.get("moods"),
            "themes": meta.get("themes"),
            "summary": meta.get("summary"),
        })
    return poems


def pretty_print(results: List[Dict]):
    for i, poem in enumerate(results, start=1):
        print(f"\n=== Result {i} (score: {poem['score']:.3f}) ===")
        print(f"Title : {poem.get('title')}")
        print(f"Author: {poem.get('author')}")
        print(f"URL   : {poem.get('url')}")
        if poem.get("moods"):
            print(f"Moods : {', '.join(poem['moods'])}")
        if poem.get("themes"):
            print(f"Themes: {', '.join(poem['themes'])}")
        if poem.get("summary"):
            print(f"Summary: {poem['summary']}")


if __name__ == "__main__":
    user_query = input("Describe what you're looking for (feeling, mood, etc.): ")
    k = input("How many poems? (default 3): ").strip()
    k = int(k) if k else 3

    results = search_poems(user_query, top_k=k)
    if not results:
        print("No results found (which would be surprising).")
    else:
        pretty_print(results)

