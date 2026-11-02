
import pandas as pd
import os

# Manually reviewed labels: c = correct, w = wrong
rag_labels = {
    "P1": "w", "P2": "c", "P3": "c", "P4": "c",
    "P5": "c", "P6": "c", "P7": "c", "P8": "c",
    "C1": "w", "C2": "c", "L7": "c", "D4": "c",
    "D10": "c", "D11": "c", "D12": "c",
    "D14": "c", "D15": "c", "D21": "w"
}

crag_labels = {
    "P1": "c", "P2": "c", "P3": "c", "P4": "c",
    "P5": "c", "P6": "c", "P7": "c", "P8": "w",
    "C1": "w", "C2": "c",
    "L1": "c", "L2": "c", "L3": "c", "L4": "c",
    "L5": "c", "L6": "c", "L7": "c",
    "M1": "c", "M2": "c", "M3": "c",
    "D3": "w", "D4": "c", "D6": "c", "D7": "c",
    "D8": "c", "D9": "c", "D10": "c", "D11": "c",
    "D12": "c", "D14": "c", "D15": "c",
    "D16": "c", "D21": "w"
}

systems = [
    ("rag_results.csv", rag_labels),
    ("crag_results.csv", crag_labels)
]

for filename, labels in systems:

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue

    df = pd.read_csv(filename)
    df["label"] = df["label"].fillna("").astype(str)

    # Update manually graded labels
    for question_id, label in labels.items():
        df.loc[df["id"] == question_id, "label"] = label

    # Exclude C3 from both system evaluations
    df = df[df["id"] != "C3"]

    df.to_csv(filename, index=False)

    print(f"\n{filename}")
    print("Correct:", (df["label"] == "c").sum())
    print("Wrong:", (df["label"] == "w").sum())
    print("Refused:", (df["label"] == "r").sum())
    print("Saved successfully!")

print("\nBoth systems have been graded.")
