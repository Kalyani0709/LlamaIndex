import os
import json
import re

INPUT_DIR = "data/parsed"
OUTPUT_FILE = "data/chunked.json"

MAX_CHUNK_WORDS = 200


def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def split_blocks(text):
    """
    Keeps headings WITH content
    """
    lines = text.split("\n")
    
    blocks = []
    current_block = []

    for line in lines:
        if line.startswith("#"):
            if current_block:
                blocks.append("\n".join(current_block).strip())
            current_block = [line]  # keep heading
        else:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block).strip())

    return blocks


def merge_small_blocks(blocks):
    """
    Merge tiny blocks (like nav items) together
    """
    merged = []
    buffer = ""

    for block in blocks:
        word_count = len(block.split())

        if word_count < 20:
            buffer += "\n" + block
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(block)

    if buffer:
        merged.append(buffer.strip())

    return merged


def split_large_block(block):
    words = block.split()
    chunks = []

    for i in range(0, len(words), MAX_CHUNK_WORDS):
        chunk = " ".join(words[i:i + MAX_CHUNK_WORDS])
        chunks.append(chunk)

    return chunks


def extract_heading(block):
    first_line = block.split("\n")[0]
    return first_line.replace("#", "").strip()


def run():
    final_chunks = []

    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".md"):
            continue

        path = os.path.join(INPUT_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            content = clean_text(f.read())

        # Step 1: split properly
        blocks = split_blocks(content)

        # Step 2: merge tiny junk
        blocks = merge_small_blocks(blocks)

        for block in blocks:
            heading = extract_heading(block)

            word_count = len(block.split())

            if word_count > MAX_CHUNK_WORDS:
                sub_chunks = split_large_block(block)

                for sub in sub_chunks:
                    final_chunks.append({
                        "text": sub,
                        "metadata": {
                            "source": file,
                            "heading": heading
                        }
                    })
            else:
                final_chunks.append({
                    "text": block,
                    "metadata": {
                        "source": file,
                        "heading": heading
                    }
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2)

    print(f"✅ Created {len(final_chunks)} chunks")


if __name__ == "__main__":
    run()