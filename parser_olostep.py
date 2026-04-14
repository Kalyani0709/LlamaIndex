import json

INPUT_FILE = "data/olostep_raw.json"
OUTPUT_FILE = "data/markdown.json"

def run():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = []

    for item in data:
        markdown = item.get("markdown", "")  # 👈 key from Olostep
        source = item.get("url", "")

        if not markdown.strip():
            continue

        parsed.append({
            "source": source,
            "content": markdown
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    print(f"✅ Parsed {len(parsed)} markdown docs")

if __name__ == "__main__":
    run()