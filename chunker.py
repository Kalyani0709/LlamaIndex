import os
import json
import re

OUTPUT_FILE = "data/chunked.json"

CHUNK_SIZE = 800
OVERLAP = 120


def split_sections(text):
    sections = re.split(r"\n##+ ", text)
    return [s.strip() for s in sections if s.strip()]


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


def run():
    final_chunks = []

    for file in os.listdir("data/markdown"):
        if not file.endswith(".md"):
            continue

        path = os.path.join("data/markdown", file)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = split_sections(content)

        for sec in sections:
            title = sec.split("\n")[0]

            if len(sec.split()) < CHUNK_SIZE:
                final_chunks.append({
                    "text": sec,
                    "metadata": {"source": file, "heading": title}
                })
            else:
                chunks = chunk_text(sec)

                for c in chunks:
                    final_chunks.append({
                        "text": c,
                        "metadata": {"source": file, "heading": title}
                    })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2)

    print(f"✅ Created {len(final_chunks)} chunks")


if __name__ == "__main__":
    run()