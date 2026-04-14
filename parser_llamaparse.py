import json
from dotenv import load_dotenv
import os
from llama_index.readers.web import OlostepWebReader

load_dotenv()

API_KEY = os.getenv("OLOSTEP_API_KEY")

if not API_KEY:
    raise ValueError("❌ Missing OLOSTEP_API_KEY in .env")

RAW_URLS_FILE = "data/crawled_files.json"
OUTPUT_FILE = "data/parsed_olostep.json"


def load_urls():
    with open(RAW_URLS_FILE) as f:
        data = json.load(f)

    return [item["url"] for item in data]


def parse():
    urls = load_urls()

    print(f"🔹 Total URLs: {len(urls)}")

    reader = OlostepWebReader(
        api_key=API_KEY,
        mode="scrape"
    )

    parsed = []

    for url in urls:
        try:
            docs = reader.load_data(url=url)

            for doc in docs:
                parsed.append({
                    "source": url,
                    "content": doc.text   # already clean markdown
                })

            print("✅ Parsed:", url)

        except Exception as e:
            print("❌ Failed:", url, e)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    print(f"\n✅ Done. Parsed {len(parsed)} entries")


if __name__ == "__main__":
    parse()