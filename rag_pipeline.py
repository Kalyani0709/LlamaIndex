import os
import re
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ------------------ CONFIG ------------------
COLLECTION_NAME = "mopar_collection"

qdrant = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-mpnet-base-v2")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=30
)

# ------------------ RETRIEVE ------------------
def retrieve(query, top_k=5):
    vector = model.encode(query).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k
    ).points

    return results


# ------------------ BUILD CONTEXT ------------------
def build_context(results):
    context_blocks = []

    for r in results[:3]:   # balanced
        text = r.payload.get("content", "").strip()
        if text:
            context_blocks.append(text)

    context = "\n\n".join(context_blocks)

    print("\n===== CONTEXT =====\n", context)
    print("\n===============================================================================================\n")

    return context


# ------------------ GENERATE (OPENAI) ------------------
def generate_answer(query, context):
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # 🔥 fast + accurate
        messages=[
            {
                "role": "system",
                "content": """You are a strict extraction assistant.

Rules:
- If the answer exists in the context → return it EXACTLY
- Do NOT rephrase
- Do NOT summarize
- Do NOT add extra explanation
- Do NOT mention sources or chunks
- Do NOT change numbers

If multiple answers exist:
- Return the most relevant one

If answer is not found:
- Return exactly: I don't know
"""
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}
"""
            }
        ],
        temperature=0.0   # 🔥 critical for accuracy
    )

    return response.choices[0].message.content.strip()


# ------------------ MAIN ------------------
def ask(query):
    results = retrieve(query)

    context = build_context(results)

    answer = generate_answer(query, context)

    return answer, results