"""Auto-grader v2.  Fills the 'label' column with 'c' ONLY when it is objectively safe, lists the rest for you.
Changes over auto_grade.py:  dates/source notes inside the gold ("Oct 22, 2026", "(RBI home page, ...)") are ignored
(they made correct rate answers look unmatched), and answers that hedge ("conflicting", "cannot be determined",
"could not find") are never auto-marked correct, even if they mention the right number.
   python auto_grade_v2.py results_dev_hybrid.csv                 numeric rule, then LLM judge for the rest
   python auto_grade_v2.py results_dev_hybrid.csv --no-judge      numeric rule only
   python auto_grade_v2.py results_dev_hybrid.csv --judge-model <other groq model>   (a DIFFERENT model than the generator is fairer)

LLM judge: compares answer with the GOLD answer only (it never sees the PDF). Rows it labels get label_by=llm so you can
say so in the report and spot-check a few with label_tool.py."""
import csv, re, sys, time
print("auto_grade v2.1 (numeric + LLM judge)")
PATH = sys.argv[1] if len(sys.argv) > 1 else "results_raw.csv"
TOL = 0.01
M = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
DATE_RX = [re.compile(p, re.I) for p in (
    rf"(?<!\d)\d{{1,2}}(?:st|nd|rd|th)?\s+{M}\b\.?,?\s*(?:\d{{4}})?",
    rf"\b{M}\b\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b,?\s*(?:\d{{4}})?",
    rf"\b{M}\b\.?\s*\d{{4}}\b",
    r"\b(?:q[1-4]|fy\s?\d{2,4}|h[12])\b",
)]
HEDGE = re.compile(r"conflict|cannot be determined|can't be determined|could not find|could not verify|not enough|unable to|not verified|unclear", re.I)
NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(lakh|lac|crore|cr|%|percent|per cent)?", re.I)
MULT = {"lakh": 1e5, "lac": 1e5, "crore": 1e7, "cr": 1e7}

def strip_gold(g):
    g = re.sub(r"\([^)]*\)", " ", g or "")
    for rx in DATE_RX:
        g = rx.sub(" ", g)
    return g

def nums(text):
    out = []
    for raw, unit in NUM.findall(text or ""):
        try: v = float(raw.replace(",", ""))
        except ValueError: continue
        out.append(v * MULT.get((unit or "").lower(), 1))
    return out

def close(a, b):
    return abs(a - b) <= TOL * max(abs(a), abs(b), 1e-9) or abs(a - b) < 1e-9

def match(gold, answer):
    g = [v for v in nums(strip_gold(gold)) if not (1900 <= v <= 2100)]
    if not g:
        return None
    if HEDGE.search(answer or ""):
        return False
    ans = nums(re.sub(r"\[\d+\]|【\d+】", " ", answer or ""))
    return all(any(close(x, y) for y in ans) for x in g)

JUDGE_SYSTEM = (
    "You are a strict grader for a finance question-answering test. You get a QUESTION, the GOLD answer (the truth) and "
    "a model ANSWER. Reply with ONE line: a letter, a bar, and a reason of at most 12 words.\n"
    "C = the ANSWER contains the main facts of the GOLD answer and states nothing that contradicts it (extra correct detail is fine; "
    "different wording is fine; for rates and figures the values must be the same; a different date on the gold is fine if the values match).\n"
    "W = a key fact or number is wrong or missing, the answer is about an older period than the gold, it contradicts the gold, "
    "or it answers when the gold says to refuse.\n"
    "R = the ANSWER only says it cannot find / cannot determine / cannot help and gives no answer. "
    "If the GOLD says the question must be refused, a refusal is R and anything else is W.\n"
    "Be strict. If unsure between C and W choose W."
)

def judge(question, gold, answer, model):
    from llm import chat
    prompt = f"QUESTION: {question}\nGOLD: {gold}\nANSWER: {(answer or '').replace(chr(8239), ' ')[:1800]}"
    for attempt in range(3):
        try:
            out = chat(prompt, system=JUDGE_SYSTEM, temperature=0.0, model=model)
            m = re.match(r"\s*([CWR])\b\s*[|:\-]?\s*(.*)", out.strip(), re.I)
            if m:
                return m.group(1).lower(), m.group(2).strip()
        except Exception as e:
            print("   judge error:", str(e)[:80]); time.sleep(6 * (attempt + 1))
    return "", "judge failed"

args = sys.argv[1:]
use_judge = "--no-judge" not in args
jm = args[args.index("--judge-model") + 1] if "--judge-model" in args else None

rows = list(csv.DictReader(open(PATH, newline="", encoding="utf-8-sig")))
fields = list(rows[0].keys())
if "label_by" not in fields:
    fields.append("label_by")
auto = 0
for r in rows:
    r.setdefault("label_by", "")
    if (r.get("label") or "").strip() or r.get("auto_label") == "r":
        continue
    if r.get("answerable") == "yes" and match(r.get("gold", ""), r.get("answer", "")):
        r["label"], r["label_by"] = "c", "numeric"
        auto += 1

judged = 0
if use_judge:
    for r in rows:
        if (r.get("label") or "").strip() or r.get("auto_label") == "r":
            continue
        lab, why = judge(r["question"], r.get("gold", ""), r.get("answer", ""), jm)
        if lab:
            r["label"], r["label_by"] = lab, "llm"
            judged += 1
            print(f"  {r['id']:<4}{r['system']:<5} -> {lab}   {why}")
        time.sleep(1.5)

with open(PATH, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
left = [r for r in rows if not (r.get("label") or "").strip() and r.get("auto_label") != "r"]
print(f"numeric-labelled: {auto} | LLM-judged: {judged} | still ungraded: {len(left)}")
print("Next: python score.py", PATH)