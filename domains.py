"""Single source of truth for verified sources. Used by web_search.py, the UI badges and data/sources.csv checks."""
import csv
import os
import re
from urllib.parse import urlparse

# Exact hosts only (no wildcards). A new subdomain must be added here on purpose.
VERIFIED_HOSTS = {
    # government and regulators
    "rbi.org.in": "official", "rbidocs.rbi.org.in": "official",
    "sebi.gov.in": "official",
    "incometax.gov.in": "official", "incometaxindia.gov.in": "official",
    "indiabudget.gov.in": "official", "finmin.nic.in": "official",
    "mospi.gov.in": "official", "esankhyiki.mospi.gov.in": "official",
    "pib.gov.in": "official", "static.pib.gov.in": "official",
    "pfrda.org.in": "official", "indiapost.gov.in": "official", "nsiindia.gov.in": "official",
    "irdai.gov.in": "official", "policyholder.gov.in": "official",
    # exchanges and industry bodies
    "nseindia.com": "exchange", "nsearchives.nseindia.com": "exchange", "bseindia.com": "exchange",
    "amfiindia.com": "industry", "portal.amfiindia.com": "industry",
    # a company's own investor pages (trusted only for that company's own reports)
    "tcs.com": "company",
}


# ---- Topic routing: which hosts may be searched for which kind of question ----
# Sample scope for now: only these topics are searched on the web. Add "tax", "retirement" ... later.
ACTIVE_TOPICS = {
    "banking",
    "mutual_funds",
    "tax",
    "economy",
    "stock_market",
    "retirement",     # <- jodo
    "insurance",      # <- jodo
}

TOPIC_HOSTS = {
    "banking": ["rbi.org.in", "rbidocs.rbi.org.in"],
    "mutual_funds": ["amfiindia.com", "portal.amfiindia.com", "sebi.gov.in"],
    "tax": ["incometaxindia.gov.in", "incometax.gov.in"],
    "economy": ["mospi.gov.in", "pib.gov.in", "static.pib.gov.in", "indiabudget.gov.in"],
    "retirement": ["pfrda.org.in", "indiapost.gov.in", "nsiindia.gov.in"],
    "insurance": ["irdai.gov.in", "policyholder.gov.in"],
    "stock_market": ["sebi.gov.in", "nseindia.com", "bseindia.com"],
}

TOPIC_WORDS = {
    "banking": r"repo|reverse repo|crr|slr|rbi|reserve bank|bank rate|msf|sdf|monetary policy|mpc|liquidity|npa|upi|banking",
    "mutual_funds": r"mutual funds?|sip|nav|elss|amc|amfi|expense ratio|fund house|index fund|etf|debt fund|equity fund",
    "tax": r"income tax|tax slab|capital gains?|ltcg|stcg|itr|tds|section 80c|deduction",
    "economy": r"gdp|inflation|cpi|economic survey|fiscal deficit|growth rate",
    "retirement": r"nps|ppf|epf|pension|provident fund|retirement",
    "insurance": r"insurance|irdai|premium|policyholder|claim",
    "stock_market": r"stock|share price|sensex|nifty|ipo|demat|nse|bse|listed",
}


def topics_for(question: str) -> list:
    q = question.lower()
    return [t for t, words in TOPIC_WORDS.items() if re.search(rf"\b(?:{words})\b", q)]


def active_hosts(question: str = "") -> list:
    """Hosts that may be searched for this question: matched topics, limited to ACTIVE_TOPICS."""
    matched = [t for t in topics_for(question) if t in ACTIVE_TOPICS] or sorted(ACTIVE_TOPICS)
    hosts = []
    for t in matched:
        hosts += [h for h in TOPIC_HOSTS[t] if h not in hosts]
    return hosts


def site_clause(question: str) -> str:
    """Add this to the web search query, for example: (site:rbi.org.in OR site:rbidocs.rbi.org.in)"""
    return "(" + " OR ".join(f"site:{h}" for h in active_hosts(question)) + ")"

def search_sites(question: str = "") -> list:
    """Top-level sites for 'site:' searches (a site: search already covers its subdomains)."""
    hosts = active_hosts(question)
    return [h for h in hosts if not any(h != p and h.endswith("." + p) for p in hosts)]

def is_searchable(url: str, question: str = "") -> bool:
    """True only if the result is on a verified host that is active for this question."""
    host = _host(url)
    return host in VERIFIED_HOSTS and host in active_hosts(question)


def _host(url: str):
    """Hostname of an https URL, or None. urlparse lower-cases it and drops user info and port."""
    try:
        p = urlparse(url.strip())
    except (ValueError, AttributeError):
        return None
    if p.scheme != "https" or not p.hostname:
        return None
    host = p.hostname
    return host[4:] if host.startswith("www.") else host


def is_verified(url: str) -> bool:
    return _host(url) in VERIFIED_HOSTS


def tier_of_source(source: str) -> str:
    """Works on pipeline source labels like 'https://host/x.pdf p.8' or 'file.pdf p.3 (table)'."""
    first = source.split(" ")[0]
    if not first.lower().startswith("http"):
        return "local"                       # a PDF from your own data/ folder
    host = _host(first)
    return VERIFIED_HOSTS.get(host, "unverified")


def check_manifest(path: str = os.path.join("data", "sources.csv")) -> int:
    """data/sources.csv has columns: file,url. Every PDF in data/ should come from a verified host."""
    bad = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ok = is_verified(row["url"])
            bad += not ok
            print(("OK        " if ok else "NOT VERIFIED"), row["file"], "|", row["url"][:80])
    print(f"\n{bad} file(s) not from a verified host.")
    return bad


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "manifest":
        raise SystemExit(1 if check_manifest() else 0)
    cases = [
        ("https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx?year=2026", True),
        ("https://rbidocs.rbi.org.in/rdocs/x.pdf", True),
        ("https://Indiapost.gov.in/VAS/DOP_PDFFiles/Savings%20Bank/Public%20Provident%20Fund%20Scheme%202019%20English.pdf", True),
        ("https://static.pib.gov.in/WriteReadData/x.pdf", True),
        ("https://pfrda.org.in/documents/33652/203559/HoNPSS+2024-25.pdf", True),
        ("https://systemhealth.rbi.org.in/index.html", False),          # subdomain not on the list
        ("https://rbi.org.in.evil.com/x.pdf", False),                    # look-alike
        ("https://notrbi.org.in/x.pdf", False),                          # look-alike
        ("https://rbi.org.in@evil.com/x.pdf", False),                    # user-info trick
        ("http://www.rbi.org.in/x", False),                              # not https
        ("https://zfunds.in/article/public-provident-fund-ppf", False),  # blog
        ("not a url", False),
        ("", False),
    ]
    fails = 0
    for url, want in cases:
        got = is_verified(url)
        fails += got != want
        print("OK  " if got == want else "FAIL", want, "|", url[:70])
    print("tiers:", tier_of_source("https://www.indiabudget.gov.in/x.pdf p.8"),
          tier_of_source("ITR-2-2026-Eng.pdf p.8 (table)"),
          tier_of_source("https://blog.example.com/a p.1"))
    qs = [("What is the current RBI repo rate?", "banking"), ("What are the current CRR and SLR?", "banking"),
          ("What is the ELSS lock-in period?", "mutual_funds"), ("What is a mutual fund?", "mutual_funds"),
          ("What is the long term capital gains tax rate?", "tax"), ("How has GDP growth changed?", "economy")]
    for q, want in qs:
        got = topics_for(q)
        fails += want not in got
        print("OK  " if want in got else "FAIL", want, "<-", got, "|", q)
    print(site_clause("What is the current RBI repo rate?"))
    print(site_clause("What is the ELSS lock-in period?"))
    print(site_clause("How has GDP growth changed?"), "(economy not active yet: falls back to active topics)")
    chk = [("https://www.rbi.org.in/Scripts/x.aspx", "What is the repo rate?", True),
           ("https://www.amfiindia.com/uploads/f.pdf", "What is the repo rate?", False),   # wrong topic for this question
           ("https://www.amfiindia.com/uploads/f.pdf", "What is a mutual fund?", True),
           ("https://zfunds.in/x", "What is a mutual fund?", False)]
    for url, q, want in chk:
        got = is_searchable(url, q)
        fails += got != want
        print("OK  " if got == want else "FAIL", want, "| searchable:", url[:45], "|", q)
    print("failures:", fails)