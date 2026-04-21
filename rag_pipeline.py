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
    "content": """You are a context-based question answering assistant.

Rules:
- Use ONLY the given context
- Do NOT add external knowledge

--------------------------------
FAQ MODE (STRICT EXTRACTION)
--------------------------------
- If the context contains a question that matches the user query:
    → Return EVERYTHING under that question
    → Do NOT remove any sentences
    → Do NOT rewrite
    → Preserve exact wording and formatting
    → Include all paragraphs until the next heading (#)

--------------------------------
SECTION MODE (REWRITE MODE)
--------------------------------
- If the content is NOT an FAQ:
    → DO NOT copy the text directly
    → DO NOT include headings or titles
    → DO NOT preserve original formatting

    → Convert the content into a natural, conversational answer

Smart Formatting:
- If multiple items exist → use bullet points
- If descriptive → use paragraph
- Merge content into a clean readable response

--------------------------------
GLOBAL RULES
--------------------------------
- NEVER mix FAQ and section outputs
- NEVER output raw document structure for sections
- NEVER repeat the question
- Focus only on relevant content

If answer not found:
- Respond briefly and naturally
- Do NOT say "I don't know"
- Say something like:
  "I don’t have that information right now."
  OR
  "I’m unable to find that in the current data."
- If user asks for a file/PDF:
  "I’m unable to share that right now."
"""
},
            {
                "role": "user",
                "content": f"{context}\n\nAnswer this question:\n{query}"
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