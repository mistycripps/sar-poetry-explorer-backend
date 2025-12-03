import os
import json
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from tqdm import tqdm

# -------------------- CONFIG & SETUP --------------------

# Load environment variables from .env in this folder
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sar-poetry")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing from .env")

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Path to your JSONL file
JSONL_PATH = "sar_poems_second_reformat.jsonl"

# Embedding model and batch size
EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE = 64  # how many poems to embed in each batch


# -------------------- HELPER FUNCTIONS --------------------

def load_poems(path: str) -> List[Dict]:
    """Load a JSONL file where each line is one poem dict."""
    poems: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            poems.append(json.loads(line))
    return poems


def build_text_to_embed(poem: Dict) -> str:
    """
    Build the text we send to the embedding model.
    Adjust field names here if your JSONL structure is slightly different.
    """
    title = poem.get("title", "")
    author = poem.get("author", "")

    # This assumes your JSONL has 'content_summary'; fallback to 'summary' if not
    summary = poem.get("content_summary", poem.get("summary", ""))

    themes = poem.get("themes", [])
    moods = poem.get("moods", [])
    emotional_keywords = poem.get("emotional_keywords", [])

    themes_str = ", ".join(themes) if isinstance(themes, list) else str(themes)
    moods_str = ", ".join(moods) if isinstance(moods, list) else str(moods)
    keywords_str = ", ".join(emotional_keywords) if isinstance(emotional_keywords, list) else str(emotional_keywords)

    pieces = [
        f"Title: {title}",
        f"Author: {author}",
        f"Summary: {summary}",
        f"Themes: {themes_str}",
        f"Moods: {moods_str}",
        f"Emotional keywords: {keywords_str}",
    ]
    return "\n".join(pieces)


def get_poem_id(poem: Dict, idx: int) -> str:
    """
    Choose a stable ID for each poem in Pinecone.
    Prefer an explicit poem_id if present; otherwise fall back to index.
    """
    if "poem_id" in poem and poem["poem_id"]:
        return str(poem["poem_id"])
    if "id" in poem and poem["id"]:
        return str(poem["id"])
    return f"poem-{idx}"


# -------------------- MAIN EMBEDDING LOGIC --------------------

def main():
    print(f"Loading poems from {JSONL_PATH} ...")
    poems = load_poems(JSONL_PATH)
    print(f"Loaded {len(poems)} poems.")

    # Process in batches so we don't send everything at once
    for start in tqdm(range(0, len(poems), BATCH_SIZE), desc="Embedding & upserting"):
        batch = poems[start : start + BATCH_SIZE]

        texts = [build_text_to_embed(p) for p in batch]
        ids = [get_poem_id(p, start + i) for i, p in enumerate(batch)]

        # Call OpenAI embeddings for this batch
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )

        # Prepare vectors for Pinecone
        vectors = []
        for poem, vec_id, emb in zip(batch, ids, response.data):
            metadata = {
                "title": poem.get("title", ""),
                "author": poem.get("author", ""),
                "url": poem.get("url", ""),
                "summary": poem.get("content_summary", poem.get("summary", "")),
                "themes": poem.get("themes", []),
                "moods": poem.get("moods", []),
            }

            vectors.append(
                {
                    "id": vec_id,
                    "values": emb.embedding,
                    "metadata": metadata,
                }
            )

        # Upsert batch into Pinecone index
        index.upsert(vectors=vectors)

    print("✅ Finished embedding and upserting all poems to Pinecone.")
    print(f"Index name: {PINECONE_INDEX_NAME}")


if __name__ == "__main__":
    main()

