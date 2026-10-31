"""Runs the evaluation set through normal RAG and CRAG and saves the raw answers to results_raw.csv.

Usage:
  python evaluate.py                        all questions, both systems
  python evaluate.py --systems crag         only CRAG (cheaper)
  python evaluate.py --types live,safety    only some question types
  python evaluate.py --limit 5              only the first 5 questions

Progress is saved after every answer. If you hit a rate limit, stop and run it again later:
rows that are already finished are skipped.
"""
import argparse
import csv
import json
import os
import time
import config

from pipeline import crag, baseline_rag

CSV_PATH = "results_raw.csv"
FIELDS = ["id", "type", "answerable", "system", "question", "gold", "action", "answer",
          "sources", "official_only", "seconds", "auto_label", "label", "source_ok"]
REFUSAL_TEXT = "could not find"


def official_only(sources: list[str]) -> str:
    """'no_web' = only local PDFs, 'yes' = every web source is an official domain, 'NO' = a bug."""
    urls = [s.split(" ")[0] for s in sources if s.startswith("http")]
    if not urls:
        return "no_web"
    try:
        from web_search import is_verified
    except Exception:
        return "unknown"
    return "yes" if all(is_verified(u) for u in urls) else "NO"


def load_done() -> set:
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done.add((row["id"], row["system"]))
    return done


def append_row(row: dict):
    is_new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_one(system: str, question: str):
    start = time.time()
    if system == "crag":
        result = crag(question)
        action = result["action"]
    else:
        result = baseline_rag(question)
        action = "direct"
    return result["answer"], action, result.get("sources", []), round(time.time() - start, 1)


def main():
    global CSV_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="rag,crag")
    parser.add_argument("--types", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-file", default="eval_set.json")
    parser.add_argument("--out", default="results_raw.csv")
    parser.add_argument("--retrieval", default="", choices=["", "dense", "hybrid"])
    args = parser.parse_args()

    CSV_PATH = args.out
    if args.retrieval:
        config.RETRIEVAL_MODE = args.retrieval
    print(f"Eval file: {args.eval_file} | output: {CSV_PATH} | CRAG retrieval: {getattr(config, 'RETRIEVAL_MODE', 'dense')}")

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    wanted_types = {t.strip() for t in args.types.split(",") if t.strip()}
    items = json.load(open(args.eval_file, encoding="utf-8"))
    if wanted_types:
        items = [i for i in items if i["type"] in wanted_types]
    if args.limit:
        items = items[:args.limit]

    done = load_done()
    for n, item in enumerate(items, 1):
        for system in systems:
            if (item["id"], system) in done:
                continue
            print(f"[{n}/{len(items)}] {item['id']} ({item['type']}) {system}: {item['question']}")
            try:
                answer, action, sources, seconds = run_one(system, item["question"])
            except Exception as e:
                message = str(e)
                print(f"   failed: {message[:200]}")
                if "429" in message or "rate limit" in message.lower():
                    print("Rate limit reached. Progress is saved. Run the same command again later.")
                    return
                continue
            append_row({
                "id": item["id"], "type": item["type"],
                "answerable": "yes" if item["answerable"] else "no",
                "system": system, "question": item["question"], "gold": item["gold"],
                "action": action, "answer": answer, "sources": " ; ".join(sources),
                "official_only": official_only(sources) if system == "crag" else "no_web",
                "seconds": seconds,
                "auto_label": "r" if (action == "REFUSED" or REFUSAL_TEXT in answer.lower()) else "",
                "label": "", "source_ok": "",
            })
            time.sleep(3)                         # stay under the free-tier rate limits

    print(f"\nDone. Open {CSV_PATH}, fill the 'label' column (c / w / r), then run: python score.py")

if __name__ == "__main__":
    main()