
"""Metrics for CRAG Finance: Paper 2 evaluation.

Labels:
  c = correct, w = wrong, r = refused

source_ok:
  y = cited sources support the answer
  n = cited sources do not support the answer
"""

import csv
import sys
from collections import Counter, defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "results_raw.csv"


def norm(value):
    return str(value or "").strip().lower()


def final_label(row):
    label = norm(row.get("label"))

    if label in ("c", "w", "r"):
        return label

    if norm(row.get("auto_label")) == "r":
        return "r"

    return ""


def is_answerable(row):
    return norm(row.get("answerable")) == "yes"


def metrics(rows):
    n = len(rows)
    refused = sum(final_label(row) == "r" for row in rows)
    correct = sum(final_label(row) == "c" for row in rows)
    answered = n - refused

    return {
        "N": n,
        "U": refused,
        "C": correct,
        "accuracy": correct / n if n else 0,
        "reliability": correct / answered if answered else 0,
        "hallucination": (answered - correct) / n if n else 0,
    }


def pct(value):
    return f"{value:.1%}"


def main():
    try:
        with open(PATH, newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
    except FileNotFoundError:
        print(f"ERROR: File not found: {PATH}")
        print("Usage: python score.py crag_results.csv")
        sys.exit(1)
    except (OSError, csv.Error) as error:
        print(f"ERROR reading CSV: {error}")
        sys.exit(1)

    if not rows:
        print(f"ERROR: No rows found in {PATH}")
        sys.exit(1)

    required = {"system", "answerable", "label"}
    missing = required - set(rows[0].keys())
    if missing:
        print("ERROR: Missing CSV columns:", ", ".join(sorted(missing)))
        sys.exit(1)

    ungraded = [row for row in rows if not final_label(row)]
    if ungraded:
        ids = sorted({row.get("id") or "?" for row in ungraded})
        print(
            f"WARNING: {len(ungraded)} ungraded row(s) excluded: "
            + ", ".join(ids)
        )

    graded = [row for row in rows if final_label(row)]
    by_system = defaultdict(list)

    for row in graded:
        system = norm(row.get("system")) or "unknown"
        by_system[system].append(row)

    if not by_system:
        print("ERROR: No rows have valid labels (c, w, r).")
        sys.exit(1)

    print(f"\nFile: {PATH}")
    print(f"Total rows: {len(rows)} | Graded: {len(graded)}")

    # Answerable-question metrics
    print("\n=== ANSWERABLE QUESTIONS (PAPER 2) ===")
    print(
        f"{'System':<10}{'N':>5}{'Refused':>10}{'Correct':>10}"
        f"{'Accuracy':>12}{'Reliability':>14}{'Halluc.':>11}"
    )

    for system, system_rows in sorted(by_system.items()):
        selected = [r for r in system_rows if is_answerable(r)]
        m = metrics(selected)

        print(
            f"{system:<10}{m['N']:>5}{m['U']:>10}{m['C']:>10}"
            f"{pct(m['accuracy']):>12}{pct(m['reliability']):>14}"
            f"{pct(m['hallucination']):>11}"
        )

    # Accuracy by question type
    print("\n=== ACCURACY BY QUESTION TYPE ===")
    question_types = sorted({
        norm(r.get("type"))
        for r in graded
        if is_answerable(r) and norm(r.get("type"))
    })

    systems = sorted(by_system)
    print(f"{'Type':<12}" + "".join(f"{s:>18}" for s in systems))

    for question_type in question_types:
        line = f"{question_type:<12}"

        for system in systems:
            selected = [
                r for r in by_system[system]
                if is_answerable(r)
                and norm(r.get("type")) == question_type
            ]
            m = metrics(selected)
            result = (
                f"{m['C']}/{m['N']} ({pct(m['accuracy'])})"
                if m["N"] else "N/A"
            )
            line += result.rjust(18)

        print(line)

    # Unanswerable / safety questions
    print("\n=== REFUSAL PERFORMANCE (UNANSWERABLE QUESTIONS) ===")

    for system, system_rows in sorted(by_system.items()):
        selected = [
            r for r in system_rows
            if norm(r.get("answerable")) == "no"
        ]

        if not selected:
            print(f"{system:<10} No unanswerable questions")
            continue

        refused = sum(final_label(r) == "r" for r in selected)
        print(
            f"{system:<10} Correct refusals: {refused}/{len(selected)} | "
            f"Not refused: {len(selected) - refused}"
        )

    # CRAG-specific source and runtime reporting
    crag_rows = by_system.get("crag", [])

    if crag_rows:
        print("\n=== CRAG SOURCE AUDIT ===")

        statuses = Counter(
            norm(r.get("official_only")) or "missing"
            for r in crag_rows
        )

        for status, count in sorted(statuses.items()):
            print(f"official_only='{status}': {count}")

        flagged = [
            r for r in crag_rows
            if norm(r.get("official_only")) == "no"
        ]

        print(f"Explicitly flagged for source review: {len(flagged)}")
        print(f"Rows marked no_web: {statuses.get('no_web', 0)}")
        print(
            "Note: 'no_web' means no web source was used; "
            "it does not itself indicate a non-official source."
        )

        if flagged:
            print("\nRows flagged for source review:")
            for r in flagged:
                print(
                    f"  {r.get('id', '?')}: {r.get('question', '')}\n"
                    f"    Sources: {r.get('sources', '')}"
                )

        checked = [
            r for r in crag_rows
            if norm(r.get("source_ok")) in ("y", "n")
        ]

        if checked:
            supported = sum(
                norm(r.get("source_ok")) == "y" for r in checked
            )
            print(
                f"Citation support: {supported}/{len(checked)} "
                "audited rows"
            )
        else:
            print("Citation support: NOT AUDITED")

        seconds = []
        for r in crag_rows:
            try:
                value = float(r.get("seconds", ""))
                if value >= 0:
                    seconds.append(value)
            except (TypeError, ValueError):
                continue

        if seconds:
            print(
                f"Average time per answer: "
                f"{sum(seconds) / len(seconds):.1f} s"
            )
        else:
            print("Average time per answer: unavailable")

        actions = Counter(
            norm(r.get("action")) or "missing" for r in crag_rows
        )
        print("Pipeline actions:", dict(actions))


if __name__ == "__main__":
    main()
