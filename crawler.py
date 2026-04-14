import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from markdownify import markdownify as md
import json

load_dotenv(override=True)

# ================= CONFIG =================
BASE_DOMAINS = os.getenv("BASE_DOMAINS").split(",")
MAX_DEPTH = int(os.getenv("MAX_DEPTH", 2))

OUTPUT_HTML_DIR = "data/html"
OUTPUT_MD_DIR = "data/markdown"

os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)
os.makedirs(OUTPUT_MD_DIR, exist_ok=True)

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
            "#", "login", "account", "cart", "javascript", "mailto"
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


def clean_html(soup):
    # ❌ remove junk
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    return soup


# ================= MAIN =================
files = []

# Step 1: Crawl
for base in BASE_DOMAINS:
    domain = urlparse(base).netloc
    print("\n🔹 Crawling domain:", domain)
    crawl(base, domain)

print("\nTotal URLs found:", len(crawled))


# Step 2: Scrape + Save
for url in crawled:
    try:
        r = session.get(url, timeout=10)

        if "text/html" not in r.headers.get("Content-Type", ""):
            continue

        soup = BeautifulSoup(r.content, "html.parser")
        soup = clean_html(soup)

        # ---------- Save HTML ----------
        filename = url.replace("https://", "").replace("http://", "").replace("/", "_")[:200]
        html_path = os.path.join(OUTPUT_HTML_DIR, filename + ".html")
        save_file(html_path, r.text)

        # ---------- Save MARKDOWN (🔥 MAIN) ----------
        markdown = md(str(soup))

        if not markdown.strip():
            continue

        md_path = os.path.join(OUTPUT_MD_DIR, filename + ".md")
        save_file(md_path, markdown)

        files.append({
            "url": url,
            "html": html_path,
            "markdown": md_path
        })

        print("✅ Saved:", filename)

    except Exception as e:
        print("❌ Error:", url, e)


# Step 3: Metadata
with open("data/crawled_files.json", "w", encoding="utf-8") as f:
    json.dump(files, f, indent=2)

print("\n📁 Total files saved:", len(files))