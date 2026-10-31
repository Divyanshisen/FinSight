"""Reads charts, graphs, tables and image-only pages of a PDF with a vision model (Groq)."""
import base64
import hashlib
import io
import os
import re
import time

import config

try:
    import pymupdf as fitz          # newer PyMuPDF
except ImportError:
    import fitz                     # older PyMuPDF

PROMPT = """You are reading one page of a finance document.
Look only at charts, graphs, tables, infographics, and any text that is inside images.
If the page has none of these, reply with exactly: NO_VISUAL
Otherwise, for each visual write:
- Type and title
- Axes, units and time period
- Main values (only numbers you can read clearly; write 'unclear' otherwise)
- Trend (rising, falling, flat, peaks, drops)
- A one or two sentence takeaway
Also transcribe any text that appears inside images.
Never invent numbers. Keep it under 200 words."""

_client = None
_use_effort = True                       # try to switch off "thinking" to save output tokens


class VisionQuotaExceeded(Exception):
    """Raised when the daily token limit of the vision model is used up."""


def _retry_after_seconds(message: str):
    """Reads 'Please try again in 9.72s' or '6m40.4s' from a rate-limit message."""
    m = re.search(r"try again in (?:(\d+)m)?\s*(?:([\d.]+)s)?", message)
    if not m or not (m.group(1) or m.group(2)):
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2) or 0)


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not found. Check your .env file.")
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def page_visual_score(page, text_len: int) -> float:
    """Higher score = the page more likely contains a chart, image or infographic."""
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    img_frac = 0.0
    try:
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            img_frac += max(0.0, x1 - x0) * max(0.0, y1 - y0) / page_area
    except Exception:
        pass
    try:
        n_draw = len(page.get_drawings())
    except Exception:
        n_draw = 0

    if img_frac < 0.02 and n_draw < 5:
        return 0.0                                   # blank or plain text page
    score = img_frac * 100 + n_draw / 10
    if text_len < config.VISION_MIN_TEXT_CHARS:
        score += 1000                                # almost no text: image-only page
    elif img_frac < config.VISION_MIN_IMAGE_FRACTION and n_draw < config.VISION_MIN_DRAWINGS:
        return 0.0                                   # text page with only small decorations
    return score


def select_visual_pages(doc, text_lengths: list[int], max_pages: int | None = None) -> list[int]:
    """Returns 0-based page indexes to send to the vision model (best pages first, capped)."""
    limit = max_pages or config.VISION_MAX_PAGES_PER_PDF
    scored = [(page_visual_score(doc[i], text_lengths[i]), i) for i in range(len(doc))]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(reverse=True)
    chosen = [i for _, i in scored[:limit]]
    return sorted(chosen)                            # back to page order


def render_page_jpeg(page, max_side: int = 1600) -> bytes:
    from PIL import Image
    pix = page.get_pixmap(dpi=config.VISION_DPI, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _clean(text: str | None) -> str:
    """Removes <think>...</think> blocks that reasoning models may include."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _ask_vision(jpeg: bytes) -> str:
    global _use_effort
    client = _get_client()
    b64 = base64.b64encode(jpeg).decode()
    for attempt in range(1, 5):
        kwargs = dict(
            model=config.VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            temperature=0,
            max_completion_tokens=config.VISION_MAX_OUTPUT_TOKENS,
        )
        if _use_effort:
            kwargs["reasoning_effort"] = "none"
        try:
            resp = client.chat.completions.create(**kwargs)
            answer = _clean(resp.choices[0].message.content)
            if not answer:
                raise ValueError("empty answer from the vision model")
            return answer
        except Exception as e:
            msg = str(e)
            if "tokens per day" in msg or "(TPD)" in msg:
                raise VisionQuotaExceeded(msg) from e
            if _use_effort and "reasoning" in msg.lower() and "429" not in msg:
                _use_effort = False              # this model does not accept the option
                print("   (retrying without reasoning_effort)")
                continue
            print(f"   vision attempt {attempt} failed: {msg[:160]}")
            if attempt == 4:
                raise
            wait = _retry_after_seconds(msg)
            time.sleep(min(wait + 1, 60) if wait else 5 * attempt)


def describe_page(page) -> dict | None:
    """Returns {'description': str, 'image': path} or None if the page has no visual content.
    Results are cached on disk, so re-running ingest does not call the model again."""
    jpeg = render_page_jpeg(page)
    key = hashlib.sha1(jpeg + config.VISION_MODEL.encode()).hexdigest()[:24]
    os.makedirs(config.VISION_CACHE_DIR, exist_ok=True)
    os.makedirs(config.IMAGE_DIR, exist_ok=True)
    cache_path = os.path.join(config.VISION_CACHE_DIR, key + ".txt")

    if os.path.exists(cache_path):
        desc = open(cache_path, encoding="utf-8").read()
    else:
        desc = _ask_vision(jpeg)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(desc)
        time.sleep(config.VISION_PAUSE_SECONDS)       # stay under the per-minute token limit

    if not desc.strip() or desc.strip().upper().startswith("NO_VISUAL"):
        return None
    img_path = os.path.join(config.IMAGE_DIR, key + ".jpg")
    with open(img_path, "wb") as f:
        f.write(jpeg)
    return {"description": desc.strip(), "image": img_path}