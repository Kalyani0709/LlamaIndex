import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "mopar_collection"

qdrant = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-mpnet-base-v2")


# 🔹 Retrieve (IMPROVED)
def retrieve(query, top_k=10):
    # 🔹 Step 1: Encode query
    vector = model.encode(query).tolist()

    # 🔹 Step 2: Search in Qdrant
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k
    ).points

    query_lower = query.lower()
    boosted_results = []

    # 🔹 Step 3: Filter + Boost
    for r in results:
        text = r.payload.get("content", "").strip()
        metadata = r.payload.get("metadata", {})

        heading = metadata.get("heading", "").lower()
        word_count = metadata.get("word_count", len(text.split()))

        # ❌ Skip useless tiny chunks (like "Find Tires")
        if word_count < 10:
            continue

        # ❌ Skip weak matches
        if r.score and r.score < 0.55:
            continue

        # 🔥 FAQ BOOST (very important)
        # If query matches heading → push to top
        if heading and query_lower in heading:
            r.score = 1.0

        boosted_results.append(r)

    # 🔹 Step 4: Sort by score (highest first)
    boosted_results = sorted(
        boosted_results,
        key=lambda x: x.score if x.score else 0,
        reverse=True
    )

    # 🔹 Step 5: Fallback (if everything filtered out)
    if not boosted_results:
        return results

    return boosted_results


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