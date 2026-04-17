import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
import json

from llama_parse import LlamaParse

load_dotenv(override=True)

# ================= CONFIG =================
BASE_DOMAINS = os.getenv("BASE_DOMAINS").split(",")
MAX_DEPTH = int(os.getenv("MAX_DEPTH", 2))

OUTPUT_RAW_DIR = "data/raw"
OUTPUT_PARSED_DIR = "data/parsed"

os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_PARSED_DIR, exist_ok=True)

# ================= LLAMA PARSE =================
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown"   # can also use "text"
)

# ================= SESSION =================
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
})

crawled = set()


# ================= HELPERS =================
def is_valid(url, base_domain):
    parsed = urlparse(url)

    return (
        parsed.scheme in ["http", "https"]
        and parsed.netloc.endswith(base_domain)
        and not any(x in url.lower() for x in [
            "login", "account", "cart", "javascript", "mailto"
        ])
    )


def get_links(url, base_domain):
    links = []

    try:
        r = session.get(url, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")

        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])

            if is_valid(link, base_domain):
                links.append(link)

    except Exception as e:
        print("Link error:", url, e)

    return list(set(links))


def crawl(url, base_domain, depth=0):
    if depth > MAX_DEPTH or url in crawled:
        return

    print(f"[Depth {depth}] Crawling:", url)
    crawled.add(url)

    for link in get_links(url, base_domain):
        crawl(link, base_domain, depth + 1)


def save_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ================= MAIN =================
files = []

# Step 1: Crawl
for base in BASE_DOMAINS:
    domain = urlparse(base).netloc
    print("\n🔹 Crawling domain:", domain)
    crawl(base, domain)

print("\nTotal URLs found:", len(crawled))


# Step 2: Parse with LlamaParse
for url in crawled:
    try:
        print(f"\n🔍 Parsing: {url}")

        # Save raw HTML (optional but useful)
        r = session.get(url, timeout=10)

        if "text/html" not in r.headers.get("Content-Type", ""):
            continue

        filename = url.replace("https://", "").replace("http://", "").replace("/", "_")[:200]

        raw_path = os.path.join(OUTPUT_RAW_DIR, filename + ".html")
        save_file(raw_path, r.text)

        # 🔥 LlamaParse
        html_content = r.text

        temp_file = "temp.html"

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        parsed_docs = parser.load_data(temp_file)

        if not parsed_docs:
            continue

        parsed_text = "\n\n".join([doc.text for doc in parsed_docs])

        parsed_path = os.path.join(OUTPUT_PARSED_DIR, filename + ".md")
        save_file(parsed_path, parsed_text)

        files.append({
            "url": url,
            "raw": raw_path,
            "parsed": parsed_path
        })

        print("✅ Parsed:", filename)

    except Exception as e:
        print("❌ Error:", url, e)


# Step 3: Metadata
with open("data/parsed_files.json", "w", encoding="utf-8") as f:
    json.dump(files, f, indent=2)

print("\n📁 Total files saved:", len(files))