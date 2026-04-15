import os
import json
import re

INPUT_DIR = "data/parsed"
OUTPUT_FILE = "data/chunked.json"

CHUNK_SIZE = 800
OVERLAP = 120


# ================= SPLIT BY HEADINGS =================
def split_sections(text):
    sections = re.split(r"\n#{1,6} ", text)  # supports H1-H6
    return [s.strip() for s in sections if s.strip()]


# ================= SMART CHUNKING =================
def chunk_text(text):
    words = text.split()
    chunks = []

    step = CHUNK_SIZE - OVERLAP

    for i in range(0, len(words), step):
        chunk_words = words[i:i + CHUNK_SIZE]

        if len(chunk_words) < 50:
            continue

        chunks.append(" ".join(chunk_words))

    return chunks


# ================= MAIN =================
def run():
    final_chunks = []

    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".md"):
            continue

        path = os.path.join(INPUT_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = split_sections(content)

        for sec in sections:
            lines = sec.split("\n")
            title = lines[0] if lines else "unknown"

            word_count = len(sec.split())

            # 🔥 small section → keep as is
            if word_count < CHUNK_SIZE:
                final_chunks.append({
                    "text": sec,
                    "metadata": {
                        "source": file,
                        "heading": title
                    }
                })
            else:
                chunks = chunk_text(sec)

                for c in chunks:
                    final_chunks.append({
                        "text": c,
                        "metadata": {
                            "source": file,
                            "heading": title
                        }
                    })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2)

    print(f"✅ Created {len(final_chunks)} chunks")


if __name__ == "__main__":
    run()