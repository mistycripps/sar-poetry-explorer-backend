import os
from typing import List, Dict, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

# ------------ CONFIG & CLIENTS ------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sar-poetry")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMBED_MODEL = "text-embedding-3-large"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing from .env")


client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SAR Poetry Finder API")

ALLOWED_ORIGIN_REGEX = r"https?://(localhost(:\d+)?|127\.0\.0\.1(:\d+)?|.*\.netlify\.app)$"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory conversation store: {conversation_id: [message, ...]}
conversations: Dict[str, List[Dict[str, str]]] = {}

# Track how many times a given conversation has triggered crisis safety
crisis_attempts: Dict[str, int] = {}
MAX_CRISIS_ATTEMPTS = 3



# ------------ SAFETY / CRISIS DETECTION ------------

SELF_HARM_KEYWORDS = [
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "want to die",
    "wish I were dead",
    "hurt myself",
    "self harm",
    "self-harm",
    "off myself",
    "unalive myself",
]

HARM_OTHERS_KEYWORDS = [
    "kill my",
    "kill him",
    "kill her",
    "kill them",
    "hurt my",
    "hurt him",
    "hurt her",
    "hurt them",
    "hurt someone",
    "kill someone",
    "murder",
    "beat up",
    "attack",
]




from typing import Optional

def detect_crisis_type(message: str) -> Optional[str]:
    """
    Lightweight, deterministic crisis detector for tests and safety routing.
    Returns:
        "self"   -> self-harm / suicidal content
        "others" -> intent to harm others
        None     -> no clear crisis detected
    """
    text = message.lower()
    # Normalize whitespace a bit so matching is reliable
    text = " ".join(text.split())

    # Self-harm patterns
    SELF_PATTERNS = [
        "kill myself",
        "end my life",
        "feel suicidal",
        "i am suicidal",
        "i'm suicidal",
        "wish i were dead",
        "wish i was dead",
        "hurt myself",
        "hurting myself",        # << covers your test
        "harm myself",
        "self harm",
        "self-harm",
        "off myself",
        "unalive myself",
    ]

    # Harm-others patterns
    HARM_OTHERS_PATTERNS = [
        "killing my",            # << covers your test
        "kill my",
        "kill him",
        "kill her",
        "kill them",
        "kill someone",
        "hurt my",
        "hurt him",
        "hurt her",
        "hurt them",
        "hurt someone",
        "murder",
        "beat up",
        "attack",
    ]

    if any(p in text for p in SELF_PATTERNS):
        return "self"

    if any(p in text for p in HARM_OTHERS_PATTERNS):
        return "others"

    return None




def detect_crisis_type_via_moderation(message: str) -> Optional[str]:
    """
    Second-line semantic safety check using OpenAI's moderation API.
    Only called if detect_crisis_type() returns None.
    """
    try:
        resp = client.moderations.create(
            model="omni-moderation-latest",
            input=message,
        )
        result = resp.results[0]
        cats = result.categories  # object with category booleans

        # Access categories via indexing instead of .get(...)
        def cat(name: str) -> bool:
            try:
                return bool(cats[name])
            except Exception:
                return False

        # Self-harm: any of these means "self"
        if (
            cat("self-harm")
            or cat("self-harm/intent")
            or cat("self-harm/instructions")
        ):
            return "self"

        # Harm others: violence / graphic / threatening / violent illicit
        if (
            cat("violence")
            or cat("violence/graphic")
            or cat("harassment/threatening")
            or cat("illicit/violent")
        ):
            return "others"

    except Exception as e:
        # Don't crash the app if moderation fails; just log and fall back
        print(f"[moderation error] {e}")

    return None





def build_self_harm_response() -> str:
    return (
        "I'm really sorry you're feeling this way. I’m not able to help with situations "
        "involving self-harm or thoughts of suicide, but you’re not alone in this. "
        "It might really help to reach out to someone who can support you right now.\n\n"
        "If you are in immediate danger, please contact your local emergency number. "
        "If you're in the United States, you can call or text 988 to reach the Suicide & Crisis Lifeline. "
        "If you're elsewhere, please look up your local crisis hotline or contact a trusted person in your life."
    )


def build_harm_others_response() -> str:
    return (
        "It sounds like you’re feeling extremely overwhelmed or angry, and I’m really sorry "
        "you’re going through that. I’m not able to help with anything involving harming someone else. "
        "It may help to step away from the situation and reach out to a mental health professional "
        "or someone you trust who can help you work through what you’re feeling.\n\n"
        "If you feel you might act on these thoughts, please contact your local emergency services right away."
    )









# ------------ DATA MODELS ------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class ChatRequest(BaseModel):
    message: str
    top_k: int = 3
    conversation_id: Optional[str] = None


# ------------ HELPER FUNCTIONS ------------

def embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return resp.data[0].embedding


def pinecone_search(query: str, top_k: int = 3) -> List[Dict]:
    vector = embed_text(query)
    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    poems: List[Dict] = []
    for match in result.matches:
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
            "text": meta.get("text"),
        })
    return poems



def build_recommendation_reply(
    user_message: str,
    poems: List[Dict],
    history: List[Dict[str, str]],
    ) -> str:
    """
    Ask the LLM (in SAR voice) to pick from the candidate poems
    and talk to the user in a warm, literary way.
    """

    system_prompt = (
        "You are the San Antonio Review Poetry Finder, a literary-minded guide "
        "for San Antonio Review (SAR), an online journal of poetry and prose. "
        "Readers come to you with feelings, questions, or moods, and you help "
        "them discover poems from the SAR catalog that resonate.\n\n"
        "Tone:\n"
        "- Warm, thoughtful, and grounded in the craft of poetry.\n"
        "- Clear-eyed: you acknowledge difficulty without sugarcoating.\n"
        "- You are not a therapist and do not give clinical advice; you offer art, "
        "language, and perspective.\n\n"
        "Behavior:\n"
        "- Always base recommendations ONLY on the poems you are given in this conversation.\n"
        "- Never invent poem titles, authors, or URLs.\n"
        "- When you recommend, give 1–3 poems, each with title, author, and URL.\n"
        "- In 2–4 sentences per poem, explain why it might fit the reader's emotional state "
        "or question. You can reference mood, theme, imagery, or voice.\n"
        "- It's okay to say a poem is an indirect fit (for example, it approaches the topic "
        "sideways) as long as you explain why that might still be meaningful.\n"
        "- Invite gentle refinement, e.g. ‘If this feels too intense/too soft, we can look in "
        "a slightly different direction.’"
	"Safety:\n"
        "- If a user ever expresses suicidal intent, self-harm, or a desire to harm someone else, "
        "you must NOT provide poem recommendations or engage in literary exploration. "
        "Instead, gently encourage them to seek immediate help from crisis resources or trusted people. "
        "The backend will usually handle this, but you must never override that safety behavior."

    )

    # Turn poems into a compact context block
    poem_descriptions = []
    for p in poems:
        poem_descriptions.append(
            f"ID: {p['id']}\n"
            f"Title: {p.get('title')}\n"
            f"Author: {p.get('author')}\n"
            f"URL: {p.get('url')}\n"
            f"Moods: {p.get('moods')}\n"
            f"Themes: {p.get('themes')}\n"
            f"Summary: {p.get('summary')}\n"
        )
    context_block = "\n\n---\n\n".join(poem_descriptions)

    # This is the message for this turn
    current_user_prompt = (
        f"Reader's message:\n{user_message}\n\n"
        f"Here are candidate poems from the San Antonio Review catalog:\n\n"
        f"{context_block}\n\n"
        "Please speak directly to the reader, recommending the poem(s) that feel like "
        "the best match for their situation."
    )

    # Build the full message history: system + previous turns + this turn
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    messages.extend(history)  # prior user/assistant turns
    messages.append({"role": "user", "content": current_user_prompt})

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
    )

    return response.choices[0].message.content



# ------------ ENDPOINTS ------------

@app.get("/health")
def health():
    return {"status": "ok", "message": "SAR Poetry Finder API is running."}


@app.post("/search")
def search(req: SearchRequest):
    poems = pinecone_search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": poems}




@app.post("/chat")
def chat(req: ChatRequest):
    # 1) work out which conversation we’re in
    conv_id = req.conversation_id or str(uuid4())
    history = conversations.get(conv_id, [])

    # 2) SAFETY CHECK before any embeddings / retrieval / LLM
    crisis_type = detect_crisis_type(req.message)

    # If keyword check passes, ask the moderation model as a second line of defense
    if crisis_type is None:
        crisis_type = detect_crisis_type_via_moderation(req.message)

    if crisis_type is not None:
        # Increment crisis attempts
        current_attempts = crisis_attempts.get(conv_id, 0) + 1
        crisis_attempts[conv_id] = current_attempts

        # If repeated attempts, gently but firmly stop engaging
        if current_attempts > MAX_CRISIS_ATTEMPTS:
            reply = (
                "I’m really concerned that these thoughts keep coming up. "
                "I'm not able to help with situations involving self-harm or harming others. "
                "The safest thing you can do is reach out to a crisis line, emergency services, "
                "or someone you trust who can support you right now."
            )
        else:
            if crisis_type == "self":
                reply = build_self_harm_response()
            else:
                reply = build_harm_others_response()

        # Update history with the crisis-safe message (no poems)
        history.extend(
            [
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": reply},
            ]
        )
        conversations[conv_id] = history

        # IMPORTANT: no poems, no retrieval, no LLM
        return {
            "conversation_id": conv_id,
            "message": req.message,
            "poems": [],
            "reply": reply,
        }

    # 3) Normal path: semantic search
    poems = pinecone_search(req.message, top_k=req.top_k)

    # 4) Ask the LLM, passing prior history
    reply = build_recommendation_reply(req.message, poems, history)

    # 5) Update conversation history
    history.extend(
        [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": reply},
        ]
    )
    conversations[conv_id] = history

    return {
        "conversation_id": conv_id,
        "message": req.message,
        "poems": poems,
        "reply": reply,
    }




