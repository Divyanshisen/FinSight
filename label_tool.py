"""label_tool v2
  python label_tool.py results_final.csv            interactive labelling (c/w/r) of ungraded rows
  python label_tool.py results_final.csv --audit     citation audit: fills source_ok (y/n) for CRAG answers
  python label_tool.py results_final.csv --audit --n 8      only the first 8 rows
--audit reads the cited sources itself (local PDF pages from data/, official web pages through your fetcher), then an LLM checks
if every fact/number of the answer is in that text. Rows whose source text cannot be read are skipped (source_ok stays empty).
Chart/vision-only evidence is not visible to the audit, so a few 'n' can be false alarms: eyeball those."""
import csv, glob, re, sys, time
print("label_tool v2")
args = sys.argv[1:]
PATH = args[0] if args and not args[0].startswith("--") else "results_raw.csv"
rows = list(csv.DictReader(open(PATH, newline="", encoding="utf-8-sig")))
fields = list(rows[0].keys())

def save():
    with open(PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------------ audit
def pdf_page_text(name, page):
    import pymupdf as fitz
    hits = glob.glob(f"data/**/{name}", recursive=True) + glob.glob(name)
    if not hits:
        return ""
    doc = fitz.open(hits[0])
    return doc[page - 1].get_text()[:2500] if 0 < page <= len(doc) else ""

def source_text(src, question):
    src = src.replace("(chart/visual)", "").strip()
    m = re.match(r"^(.*?\.pdf)\s*p\.(\d+)\s*$", src, re.I)
    if m and not src.lower().startswith("http"):
        return pdf_page_text(m.group(1).strip(), int(m.group(2)))
    if src.lower().startswith("http"):
        url = re.sub(r"\s+p\.\d+$", "", src)
        from web_search import fetch_strips
        strips = fetch_strips(url, False, question)
        return " ".join(s["text"] for s in strips)[:2500]
    return ""

AUDIT_SYSTEM = ("You audit citations. You get an ANSWER and the SOURCE TEXT it was built from. "
                "Reply on one line: 'Y | reason' if every fact and number in the ANSWER is stated in the SOURCE TEXT, "
                "'N | reason' if any fact or number is missing from it or contradicts it. Reason: max 12 words. Ignore citation marks like [1].")

def audit(n):
    from llm import chat
    todo = [r for r in rows if r["system"] == "crag" and r["answerable"] == "yes" and (r.get("label") or "").strip() in ("c", "w")
            and not (r.get("source_ok") or "").strip()]
    if n: todo = todo[:n]
    print(f"auditing {len(todo)} CRAG row(s)")
    done = 0
    for r in todo:
        parts = [s.strip() for s in (r["sources"] or "").split(";") if s.strip()]
        ev = []
        for s in parts[:6]:
            try:
                t = source_text(s, r["question"])
            except Exception as e:
                t = ""
            if t: ev.append(f"[{s}] {t}")
        evidence = "\n\n".join(ev)[:9000]
        if not evidence:
            print(f"  {r['id']:<4} skipped (source text not readable)")
            continue
        try:
            out = chat(f"ANSWER: {(r['answer'] or '').replace(chr(8239), ' ')[:1500]}\n\nSOURCE TEXT:\n{evidence}", system=AUDIT_SYSTEM, temperature=0.0)
        except Exception as e:
            print(f"  {r['id']:<4} LLM error: {str(e)[:60]}"); continue
        m = re.match(r"\s*([YN])\b\s*[|:\-]?\s*(.*)", out.strip(), re.I)
        if m:
            r["source_ok"] = m.group(1).lower(); done += 1
            print(f"  {r['id']:<4} -> {r['source_ok']}   {m.group(2)[:90]}")
            save()
        time.sleep(1.5)
    save()
    ok = sum(1 for r in rows if r["system"] == "crag" and r.get("source_ok") == "y")
    tot = sum(1 for r in rows if r["system"] == "crag" and r.get("source_ok") in ("y", "n"))
    print(f"\ncitation support: {ok}/{tot} audited CRAG rows. Run: python score.py {PATH}")

if "--audit" in args:
    audit(int(args[args.index("--n") + 1]) if "--n" in args else 0)
    sys.exit()

# ------------------------------------------------------------------ interactive labelling
todo = [r for r in rows if not (r.get("label") or "").strip() and r.get("auto_label") != "r"]
print(f"{len(todo)} row(s) to label in {PATH}\n")
for n, r in enumerate(todo, 1):
    print("=" * 70)
    print(f"[{n}/{len(todo)}] {r['id']} ({r['type']}) system={r['system']} action={r['action']}")
    print("Q     :", r["question"]); print("GOLD  :", r["gold"])
    print("ANSWER:", (r["answer"] or "").replace("\u202f", " ")); print("SOURCES:", (r["sources"] or "")[:300])
    while True:
        c = input("label [c/w/r/s/q]: ").strip().lower()
        if c in ("c", "w", "r", "s", "q"): break
    if c == "q": break
    if c == "s": continue
    r["label"] = c
    save()
save()
print("saved. Now run: python score.py", PATH)