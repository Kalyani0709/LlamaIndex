import os
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

# ------------------ RETRIEVE + EXPAND ------------------
def retrieve(query, top_k=5):
    vector = model.encode(query).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k
    ).points

    expanded = []
    seen_ids = set()

    for r in results:
        expanded.append(r)
        seen_ids.add(r.id)

        meta = r.payload.get("metadata", {})
        source = meta.get("source")
        chunk_id = meta.get("chunk_id")

        if chunk_id is None:
            continue

        # 🔥 FETCH NEXT CHUNK
        next_chunks, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={
                "must": [
                    {"key": "metadata.source", "match": {"value": source}},
                    {"key": "metadata.chunk_id", "match": {"value": chunk_id + 1}}
                ]
            },
            limit=1
        )

        for n in next_chunks:
            if n.id not in seen_ids:
                expanded.append(n)
                seen_ids.add(n.id)

    return expanded


# ------------------ BUILD CONTEXT ------------------
def build_context(results):
    context_blocks = []
    seen = set()

    for r in results:
        text = r.payload.get("content", "").strip()

        if text and text not in seen:
            context_blocks.append(text)
            seen.add(text)

    context = "\n\n".join(context_blocks[:5])  # limit

    print("\n===== CONTEXT =====\n", context)
    print("\n=====================================================\n")

    return context


# ------------------ GENERATE ------------------
def generate_answer(query, context):
    if not context.strip():
        return "I don't know"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a strict context extraction assistant.

Rules:
- Return the FULL relevant section from the context
- Do NOT summarize
- Do NOT skip lines
- If heading + content → return all
- Combine chunks if needed
- Do NOT add external knowledge

If answer not found:
- Return exactly: I don't know
"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0.0
    )

    return response.choices[0].message.content.strip()


# ------------------ MAIN ------------------
def ask(query):
    results = retrieve(query)
    context = build_context(results)
    answer = generate_answer(query, context)
    return answer, results