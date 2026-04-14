import os
import uuid
import json
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

# ------------------ LOAD ENV ------------------
load_dotenv()

# ------------------ EMBEDDING MODEL ------------------
model = SentenceTransformer("all-mpnet-base-v2")

# ------------------ QDRANT ------------------
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "mopar_collection"

INPUT_FILE = "data/chunked.json"


# ------------------ LOAD CHUNKS ------------------
def load_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------ EMBEDDING ------------------
def get_embedding(text):
    return model.encode(text).tolist()


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


# ------------------ ADD DOCUMENTS ------------------
def add_documents_to_qdrant(docs, collection_name, batch_size=100):
    total = len(docs)

    print(f"\n🚀 Total chunks to upload: {total}")

    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]
        points = []

        print(f"\n📦 Uploading batch {i} → {i + len(batch)}")

        for doc in batch:
            text = doc["text"]

            if not text.strip():
                continue

            embedding = get_embedding(text)

            payload = {
                "content": text,
                "source": doc.get("source", "")
            }

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload,
                )
            )

        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )

        print("✅ Batch uploaded")

    print("\n🎉 All batches uploaded successfully!")


# ------------------ MAIN ------------------
if __name__ == "__main__":
    print("Step 1: Create collection")
    create_collection(COLLECTION_NAME)

    print("\nStep 2: Load chunked data")
    docs = load_chunks()

    print(f"\nTotal chunks: {len(docs)}")

    print("\nStep 3: Add to Qdrant")
    add_documents_to_qdrant(docs, COLLECTION_NAME)

    print("\n✅ DONE")