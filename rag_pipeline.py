import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "mopar_collection"

qdrant = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-mpnet-base-v2")


# 🔹 Retrieve (IMPROVED)
import re

def retrieve(query, top_k=10):
    vector = model.encode(query).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k
    ).points

    query_lower = query.lower()
    boosted_results = []

    for r in results:
        text = r.payload.get("content", "")
        metadata = r.payload.get("metadata", {})
        heading = metadata.get("heading", "").lower()

        score = r.score or 0

        if query_lower in heading:
            score += 0.3

        if re.search(r"\d{3}[-\s]\d{3}[-\s]\d{4}", text):
            score += 0.4

        if "|" in text:
            score += 0.2

        boosted_results.append((r, score))

    boosted_results.sort(key=lambda x: x[1], reverse=True)

    return [r for r, _ in boosted_results]


# 🔹 Build context (BETTER)
def build_context(results):
    context_blocks = []

    for i, r in enumerate(results[:5]):
        text = r.payload.get("content", "").strip()
        metadata = r.payload.get("metadata", {})

        source = metadata.get("source", "")
        heading = metadata.get("heading", "")

        block = f"""
[Chunk {i+1}]
Source: {source}
Section: {heading}
{text}
"""
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    print("\n===== CONTEXT =====\n", context)
    print("\n===============================================================================================\n")

    return context


# 🔹 Generate answer (FIXED PROMPT)
def generate_answer(query, context):
    prompt = f"""
You are a precise extraction assistant.

Follow this logic STRICTLY:

IMPORTANT RULES:
- If the answer contains numbers (phone numbers, IDs, codes):
  → You MUST copy them EXACTLY from the context
  → DO NOT modify even a single digit

STEP 1:
If any chunk contains a direct answer to the question:
- Return that answer EXACTLY as written
- DO NOT change wording
- DO NOT convert into steps
- DO NOT summarize
- DO NOT add formatting
- Just copy the answer

STEP 2:
If no direct answer is found:
- Then combine relevant information from multiple chunks
- Keep it clear and structured

Rules:
- Prefer exact copying over rewriting
- Do NOT add extra knowledge
- If not found → say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2
        },
        timeout=240
    )

    return response.json()["response"].strip()


# 🔹 Main
def ask(query):
    results = retrieve(query)
    context = build_context(results)
    answer = generate_answer(query, context)

    return answer, results