"""Verified-only web retrieval. No extra API keys.
Flow: official live pages (direct) + keyless search limited to official sites -> download the page or PDF
-> read text, tables AND charts (vision) -> strips -> keep only relevant strips (evaluator).
Which hosts count as verified, and which are active for a question, lives in domains.py."""
import re

import requests
from bs4 import BeautifulSoup

import config
import vision
from domains import is_searchable, search_sites
from llm import chat
from refine import select_strips, split_strips

try:
    import pymupdf as fitz          # newer PyMuPDF
except ImportError:
    import fitz                     # older PyMuPDF

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024

REWRITE_PROMPT = """Extract at most three keywords separated by comma from the following question as a query for web search, including topic background and main intent.

question: What was HDFC Bank's net profit in Q2 FY24?
query: HDFC Bank, net profit, Q2 FY24
question: What is the current repo rate set by RBI?
query: RBI, repo rate, current
question: What is the lock-in period of ELSS mutual funds?
query: ELSS, lock-in period
question: {question}
query:"""


def rewrite_query(question: str) -> str:
    q = chat(REWRITE_PROMPT.format(question=question), model=getattr(config, "HELPER_MODEL", None))
    q = q.strip().strip('"').split("\n")[0]
    return re.sub(r"^\s*query\s*:\s*", "", q, flags=re.I).strip()       # the model sometimes repeats "query:"


# ---------------------------------------------------------------- verification
def is_verified(url: str, question: str = "") -> bool:
    """True only if the URL is https, its exact host is listed in domains.py, and that host is active for this question."""
    return is_searchable(url, question)


def is_useful_url(url: str) -> bool:
    """Skips internal search-result pages, they never contain the answer."""
    return "searchresults" not in url.lower()


def domains_for(question: str) -> list[str]:
    """Sites to search for this question (only the active topics in domains.py, for example RBI and mutual funds)."""
    return search_sites(question)[:3]


def direct_urls_for(question: str) -> list[str]:
    """Official live pages that always hold the latest value for this kind of question."""
    q = question.lower()
    urls = []
    for keywords, pages in config.DIRECT_PAGES:
        if any(k in q for k in keywords):
            urls += [u for u in pages if u not in urls and is_verified(u, question)]
    return urls


# ---------------------------------------------------------------- search
def _search_once(query: str):
    """Keyless search. Returns a list (maybe empty) or None if every search engine failed."""
    from ddgs import DDGS
    reached = False
    last_error = None
    for backend in (None, "bing", "brave"):          # different engines if one is blocked
        try:
            kwargs = {"max_results": 10}
            if backend:
                kwargs["backend"] = backend
            results = list(DDGS(timeout=10).text(query, **kwargs))
            reached = True
            if results:
                return results
        except Exception as e:
            last_error = e
    if not reached:
        print(f"   search failed for '{query}': {last_error}")
        return None
    return []


def search_verified(question: str, query: str) -> list[dict]:
    """Searches each matching official site; results from any other site are dropped."""
    found, seen = [], set()
    for domain in domains_for(question):
        results = _search_once(f"{query} site:{domain}")
        if results is None:
            break                                     # network or engine problem, stop early
        for r in results:
            url = r.get("href") or r.get("url") or ""
            if url and url not in seen and is_verified(url, question) and is_useful_url(url):
                seen.add(url)
                found.append(r)
        if len(found) >= config.WEB_TOP_K:
            break
    return found[:config.WEB_TOP_K]

def _snippet_strips(hit: dict, url: str) -> list[dict]:
    """Use a search snippet only when it comes from a verified official result."""
    if not is_verified(url) or not is_useful_url(url):
        return []

    snippet = hit.get("body", "") or hit.get("snippet", "")
    snippet = " ".join(snippet.split()).strip()

    if len(snippet) < 40:
        return []

    return [{"text": snippet, "source": url, "image": ""}]

# ---------------------------------------------------------------- download + read
DATE_NOTE = re.compile(r"((?:last updated(?: date)?|as at|as on|updated on)[:\s]+[^\n|]{0,45}?\d{4})", re.I)


def _page_date_note(text: str) -> str:
    """Finds 'As at ... 2026' or 'last updated ... 2026' so the answer can mention how fresh the page is."""
    notes = []
    for m in DATE_NOTE.finditer(text):
        note = " ".join(m.group(1).split())
        if note not in notes:
            notes.append(note)
    return "; ".join(notes[:2])


def _html_strips(html: str, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    date_note = _page_date_note(soup.get_text("\n"))
    suffix = f" (page says: {date_note})" if date_note else ""
    strips = []

    # tables: rates, prices and figures usually live here, one strip per table
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" ".join(cells))
        text = " | ".join(rows)
        if 20 <= len(text) <= 1500:
            heading = table.find_previous(["h1", "h2", "h3", "h4"])
            title = heading.get_text(" ", strip=True) + ": " if heading else ""
            strips.append({"text": title + text + suffix, "source": url, "image": ""})
        table.decompose()                               # do not read the same cells again as paragraphs

    paras = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li"])]
    strips += [{"text": p + suffix, "source": url, "image": ""} for p in paras if len(p) >= 60][:80]
    return strips


VISUAL_WORDS = ("chart", "graph", "trend", "plot", "table", "growth", "history", "over time",
                "compare", "comparison", "how has", "movement", "visual", "figure")


def wants_visual(question: str) -> bool:
    """Charts are read at question time only when the question asks for them (saves vision quota)."""
    q = question.lower()
    return any(w in q for w in VISUAL_WORDS)


def _pdf_strips(content: bytes, url: str, use_vision: bool) -> list[dict]:
    doc = fitz.open(stream=content, filetype="pdf")
    page_texts = [doc[i].get_text() for i in range(len(doc))]
    strips = []
    for i, text in enumerate(page_texts[:config.WEB_PDF_MAX_PAGES]):
        for st in split_strips(text):
            strips.append({"text": st, "source": f"{url} p.{i + 1}", "image": ""})

    if use_vision and config.USE_VISION:               # read the charts of the downloaded PDF too
        idxs = vision.select_visual_pages(doc, [len(t) for t in page_texts],
                                          max_pages=config.WEB_VISION_MAX_PAGES)
        for idx in idxs:
            try:
                result = vision.describe_page(doc[idx])
            except vision.VisionQuotaExceeded:
                print("   Daily vision limit reached, skipping charts of this PDF.")
                break
            except Exception as e:
                print(f"   vision failed on {url} p.{idx + 1}: {e}")
                continue
            if result:
                strips.append({"text": f"[Chart/visual on page {idx + 1}] {result['description']}",
                               "source": f"{url} p.{idx + 1} (chart/visual)",
                               "image": result["image"]})
    return strips


def fetch_strips(url: str, use_vision: bool = False, question: str = "") -> list[dict]:
    """Downloads one official page or PDF and turns it into strips."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        if not is_verified(resp.url, question):          # a redirect must not leave the verified hosts
            print(f"   skipped (redirected outside verified hosts): {resp.url}")
            return []
        if len(resp.content) > MAX_DOWNLOAD_BYTES:
            print(f"   skipped (too large): {url}")
            return []
        content_type = resp.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
            return _pdf_strips(resp.content, url, use_vision)
        return _html_strips(resp.text, url)
    except Exception as e:
        print(f"   download failed: {url} -> {e}")
        return []


LIVE_WORDS = re.compile(r"\b(current|currently|latest|today|now|presently|at present|as of)\b", re.I)


def web_knowledge(question: str) -> tuple[str, list[dict]]:
    """Returns (rewritten_query, k_ex). k_ex comes only from official sources."""
    if LIVE_WORDS.search(question):                      # live question + official live page: use that page only
        direct = direct_urls_for(question)
        if direct:
            live = []
            for url in direct:
                live += fetch_strips(url, False, question)
            if live:
                return "(official live page)", select_strips(question, live)
    query = rewrite_query(question)
    use_vision = wants_visual(question)

    urls, snippets = [], {}
    for url in direct_urls_for(question):               # live official pages first
        urls.append(url)
    for hit in search_verified(question, query):         # then official search results
        url = hit.get("href") or hit.get("url") or ""
        if url and url not in urls:
            urls.append(url)
            snippets[url] = hit.get("body", "")

    strips = []
    for url in urls:
        got = fetch_strips(url, use_vision, question)

        if not got and snippets.get(url):
            got = _snippet_strips(
                {"body": snippets[url]},
                url,
            )

        strips += got
    return query, select_strips(question, strips)