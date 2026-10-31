import os
from dotenv import load_dotenv

load_dotenv()

# ---- LLM ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# ---- Models ----
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EVALUATOR_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---- Paths ----
DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION = "finance_docs"

# ---- Chunking ----
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ---- Retrieval ----
TOP_K = 5

# ---- Action trigger thresholds ----
UPPER_THRESHOLD = 0.6
LOWER_THRESHOLD = 0.3

# ---- Knowledge refinement ----
STRIP_TOP_K = 5
STRIP_THRESHOLD = 0.2
SENTENCES_PER_STRIP = 2

# ---- Web search ----
WEB_TOP_K = 5

# ---- Vision (charts / graphs / image-only pages) ----
USE_VISION = True
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
VISION_DPI = 110                    # page image quality
VISION_MAX_PAGES_PER_PDF = 12       # max pages per PDF sent to the vision model
VISION_MIN_TEXT_CHARS = 200         # page with less text than this is treated as image-only
VISION_MIN_IMAGE_FRACTION = 0.08    # image covering 8%+ of the page counts as a visual
VISION_MIN_DRAWINGS = 25           # many vector lines usually means a chart
IMAGE_DIR = "page_images"           # rendered page images (shown in the UI later)
VISION_CACHE_DIR = "vision_cache"   # cached model answers, so re-ingest is free
# ---- Verified-only web retrieval (official sites only) ----
WEB_PDF_MAX_PAGES = 40          # max pages of a downloaded PDF to read as text
WEB_VISION_MAX_PAGES = 1        # max chart pages of a downloaded PDF to read with vision

VISION_MAX_OUTPUT_TOKENS = 600  # stays under the per-minute output limit
VISION_PAUSE_SECONDS = 8        # pause after each new vision call

VERIFIED_DOMAINS = [
    "rbi.org.in", "sebi.gov.in", "nseindia.com", "bseindia.com", "amfiindia.com",
    "incometax.gov.in", "incometaxindia.gov.in", "cbic-gst.gov.in", "gst.gov.in",
    "irdai.gov.in", "pfrda.org.in", "epfindia.gov.in", "indiabudget.gov.in",
    "dea.gov.in", "finmin.gov.in", "mospi.gov.in", "npci.org.in", "nism.ac.in",
]

# keywords in the question -> which official sites to search
TOPIC_DOMAINS = [
    (["repo", "rbi", "reserve bank", "inflation", "monetary", "bank", "loan", "forex", "rupee", "upi", "crr", "slr"], ["rbi.org.in", "npci.org.in"]),
    (["mutual fund", "sip", "nav", "elss", "amc"], ["amfiindia.com", "sebi.gov.in"]),
    (["tax", "itr", "tds", "capital gain", "income tax", "80c"], ["incometax.gov.in", "incometaxindia.gov.in"]),
    (["gst"], ["cbic-gst.gov.in", "gst.gov.in"]),
    (["share", "stock", "ipo", "nifty", "sensex", "nse", "bse", "dividend", "demat", "market"], ["nseindia.com", "bseindia.com", "sebi.gov.in"]),
    (["insurance", "irdai", "premium", "claim"], ["irdai.gov.in"]),
    (["nps", "pension", "pfrda", "retirement", "epf", "provident"], ["pfrda.org.in", "epfindia.gov.in"]),
    (["budget", "gdp", "economy", "economic survey", "fiscal", "deficit"], ["indiabudget.gov.in", "dea.gov.in", "mospi.gov.in"]),
]
# ---- Official pages that are fetched directly (live data, no search engine needed) ----
DIRECT_PAGES = [
    (["repo rate", "reverse repo", "policy rate", "crr", "slr", "bank rate", "mclr",
      "marginal standing", "standing deposit", "exchange rate"], ["https://rbi.org.in/home.aspx"]),
    (["cpi", "consumer price", "inflation rate", "cfpi"], ["https://www.mospi.gov.in/uploads/latestreleasesfiles/1791801549227-Press%20Release%20for%20the%20month%20of%20September,%202026.pdf"]),
]
# ---- Retrieval mode (paper 2) ----
RETRIEVAL_MODE = "dense"      # "dense" = original retrieval, "hybrid" = Multi-HyDE + BM25 + rerank
HYDE_QUERIES = 3               # use 2 while developing to save tokens
DENSE_PER_QUERY = 5            # chunks fetched per hypothetical passage
BM25_TOP_K = 15                # paper 2: K2 = 15
HYDE_CACHE_FILE = "hyde_cache.json"

# ---- Token saving ----
HELPER_MODEL = os.getenv("HELPER_MODEL", LLM_MODEL)   # model for the small helper calls