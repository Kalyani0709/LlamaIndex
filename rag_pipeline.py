import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "mopar_collection"

qdrant = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-mpnet-base-v2")


# 🔹 Retrieve
def retrieve(query, top_k=5):
    vector = model.encode(query).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k
    ).points

    return results


# 🔹 Build context (TOP 2 chunks)
def build_context(results):
    texts = []

    for r in results[:2]:
        texts.append(r.payload.get("content", ""))

    context = "\n\n".join(texts)

    print("\n===== CONTEXT =====\n", context)

    return context


# 🔹 Generate answer (STRICT)
def generate_answer(query, context):
    prompt = f"""
You are a strict extraction assistant.

Rules:
- Use ONLY the provided context
- DO NOT summarize
- DO NOT skip steps
- Return COMPLETE answer
- Keep formatting (bullets, steps)
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
            "stream": False
        },
        timeout=120
    )

    return response.json()["response"].strip()


# 🔹 Main
def ask(query):
    results = retrieve(query)
    context = build_context(results)
    answer = generate_answer(query, context)

    return answer, results