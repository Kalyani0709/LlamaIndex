import json
import re

INPUT_FILE = "data/parsed_olostep.json"
OUTPUT_FILE = "data/chunked.json"

# 🔥 CONFIG
CHUNK_SIZE = 700
OVERLAP = 120


# 🔹 Clean noisy lines
def clean_lines(text):
    lines = text.split("\n")

    cleaned = []
    for line in lines:
        line = line.strip()

        if not line:
            continue
        if len(line) < 3:
            continue
        if line.lower() in ["submit", "click here", "learn more"]:
            continue

        cleaned.append(line)

    return cleaned


# 🔹 Split by headings (##, ###)
def split_by_headings(text):
    sections = re.split(r"\n#{2,3} ", text)
    return [sec.strip() for sec in sections if sec.strip()]


# 🔥 Split into natural semantic blocks (NO hardcoding)
def split_semantic_blocks(text):
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if len(b.strip()) > 20]


# 🔥 Smart chunking with overlap
def chunk_text(text):
    words = text.split()
    chunks = []

    step = CHUNK_SIZE - OVERLAP

    for i in range(0, len(words), step):
        chunk_words = words[i:i + CHUNK_SIZE]

        # 🔥 keep even smaller chunks (important for contacts)
        if len(chunk_words) < 15:
            continue

        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks


def run():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_chunks = []

    for item in data:
        source = item.get("source", "")
        content = item.get("content", "")

        sections = split_by_headings(content)

        for sec in sections:
            lines = clean_lines(sec)

            if not lines:
                continue

            text = "\n".join(lines)

            # 🔥 Split into semantic blocks
            blocks = split_semantic_blocks(text)

            for block in blocks:

                # 🔥 VERY IMPORTANT: preserve small meaningful blocks
                if len(block.split()) < 80:
                    final_chunks.append({
                        "text": block,
                        "metadata": {
                            "source": source
                        }
                    })
                    continue

                # 🔥 Chunk larger blocks
                chunks = chunk_text(block)

                for c in chunks:
                    final_chunks.append({
                        "text": c,
                        "metadata": {
                            "source": source
                        }
                    })

    # 🔹 Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2)

    print(f"✅ Created {len(final_chunks)} chunks")
    print(f"📁 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()