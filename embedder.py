import os
import uuid
import json
import time
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

# ------------------ LOAD ENV ------------------
load_dotenv()

# ------------------ MODEL ------------------
model = SentenceTransformer("all-mpnet-base-v2")

# ------------------ QDRANT ------------------
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "mopar_collection"

INPUT_FILE = "data/chunked.json"

BATCH_SIZE = 64   # 🔥 safer
RETRY = 3         # 🔥 production safety


# ------------------ LOAD ------------------
def load_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------ CREATE COLLECTION ------------------
def create_collection(collection_name):
    if qdrant.collection_exists(collection_name):
        print("⚠️ Deleting existing collection...")
        qdrant.delete_collection(collection_name)

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    print("✅ Collection created:", collection_name)


# ------------------ UPSERT WITH RETRY ------------------
def safe_upsert(points):
    for attempt in range(RETRY):
        try:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            return
        except Exception as e:
            print(f"⚠️ Retry {attempt+1}/{RETRY} failed:", e)
            time.sleep(2)

    print("❌ Failed batch permanently")


# ------------------ ADD DOCUMENTS ------------------
def add_documents_to_qdrant(docs):
    total = len(docs)

    print(f"\n🚀 Total chunks: {total}")

    for i in range(0, total, BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]

        print(f"\n📦 Batch {i} → {i + len(batch)}")

        texts = [doc["text"] for doc in batch if doc["text"].strip()]

        # 🔥 FAST: batch embedding
        embeddings = model.encode(texts, batch_size=16).tolist()

        points = []

        for doc, emb in zip(batch, embeddings):
            text = doc["text"]

            payload = {
                "content": text,
                "metadata": {   # ✅ FIXED
                    "source": doc.get("metadata", {}).get("source", "")
                }
            }

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload=payload,
                )
            )

        # 🔥 SAFE UPSERT
        safe_upsert(points)

        print("✅ Uploaded")

    print("\n🎉 ALL DONE")


# ------------------ MAIN ------------------
if __name__ == "__main__":
    print("Step 1: Create collection")
    create_collection(COLLECTION_NAME)

    print("\nStep 2: Load chunks")
    docs = load_chunks()

    print("\nStep 3: Upload")
    add_documents_to_qdrant(docs)