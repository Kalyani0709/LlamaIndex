import re
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ------------------ CONFIG ------------------
COLLECTION_NAME = "mopar_collection"

qdrant = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-mpnet-base-v2")


# ------------------ NORMALIZE ------------------
def normalize(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower())


# ------------------ RETRIEVE ------------------
def retrieve(query, top_k=6):
    query_vector = model.encode(query).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    return results


# ------------------ SMART RERANK (NO HARDCODING) ------------------
def rerank_results(results, query):
    query_words = normalize(query).split()

    scored = []

    for r in results:
        text = r.payload.get("content", "")
        text_clean = normalize(text)

        # 🔹 1. keyword overlap
        overlap = sum(1 for w in query_words if w in text_clean)

        # 🔹 2. exact phrase match
        phrase_bonus = 10 if normalize(query) in text_clean else 0

        # 🔥 3. proximity score (VERY IMPORTANT)
        positions = []
        for w in query_words:
            idx = text_clean.find(w)
            if idx != -1:
                positions.append(idx)

        proximity_score = 0
        if len(positions) >= 2:
            distance = max(positions) - min(positions)
            proximity_score = max(0, 50 - distance) / 50  # closer words → higher score

        # 🔹 4. length penalty (avoid long noisy chunks)
        length_penalty = len(text_clean) / 1000

        # 🔹 5. numeric bonus (helps for phone, price, etc.)
        number_bonus = 1 if re.search(r'\d{3,}', text) else 0

        # 🔥 FINAL SCORE
        final_score = (
            r.score
            + (0.2 * overlap)
            + phrase_bonus
            + (0.3 * proximity_score)
            + number_bonus
            - (0.05 * length_penalty)
        )

        scored.append((final_score, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [r for _, r in scored]


# ------------------ BUILD CONTEXT (ONLY BEST) ------------------
def build_context(results):
    if not results:
        return ""

    best = results[0]
    print("\nBest raw result:", best)
    context = best.payload.get("content", "")

    print("\n===== BEST CONTEXT =====\n", context)

    return context


# ------------------ GENERATE ANSWER ------------------
def generate_answer(query, context):
    prompt = f"""
You are a factual extraction assistant.

Rules:
- DO NOT summarize
- DO NOT rewrite
- DO NOT shorten
- Return the answer EXACTLY as it appears in the context
- Preserve steps, numbering, bullets, formatting
- Include ALL relevant steps or details
- If answer not found, say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",   # ⚡ can switch to mistral for speed
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"].strip()


# ------------------ MAIN PIPELINE ------------------
def ask(query):
    # 1️⃣ Retrieve
    results = retrieve(query)

    # 2️⃣ Rerank (smart)
    results = rerank_results(results, query)

    # 3️⃣ Build context (ONLY BEST)
    context = build_context(results)

    # 4️⃣ Generate answer
    answer = generate_answer(query, context)

    return answer, results