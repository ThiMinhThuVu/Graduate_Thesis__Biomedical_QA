import os
import json

SOURCE_ORDER = [
    "google",
    "wiki",
    "crawl_wiki",
    "crawl_ner_wiki",
    "crawl_sparql_wikidata"
]

with open('data/all_answer_golden_61.json', "r", encoding="utf-8") as f:
    dict_golden = json.load(f)

def normalize_source(src):
    return src.strip().lower() if src else "unknown"


def analyze_accuracy_ignore_no_eve(json_folder):
    """
    - no evidence = short_answer == 'insufficient information'
    - chỉ tính accuracy trên các câu có evidence
    """

    stats = {
        src: {
            "total_with_eve": 0,
            "correct": 0,
            "accuracy": 0.0
        }
        for src in SOURCE_ORDER
    }

    total_questions = 0

    for fname in os.listdir(json_folder):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(json_folder, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        qidx = data["QIDX"]
        list_gol = dict_golden[qidx]['True']
        
        total_questions += 1


        source_outputs = data['1']["0"]["0"].get("source_output", {})

        for src, result in source_outputs.items():
            src = normalize_source(src)
            if src not in stats:
                continue

            short_answer = result.get("short_answer", "").strip().lower()

            # ❌ no eve
            if short_answer.lower() == "insufficient information":
                continue

            # ✅ có eve
            stats[src]["total_with_eve"] += 1
            
            for gol in list_gol:
                if short_answer.lower() == gol:
                    stats[src]["correct"] += 1
                    break

    # tính accuracy
    for src in stats:
        total = stats[src]["total_with_eve"]
        correct = stats[src]["correct"]
        stats[src]["accuracy"] = correct / total if total > 0 else 0.0

    return stats, total_questions


def print_report(stats, total_questions):
    print("\n=== ACCURACY (IGNORE NO-EVIDENCE) ===")
    print(f"Total questions: {total_questions}\n")

    for src in SOURCE_ORDER:
        s = stats[src]
        print(
            f"{src:25s} | "
            f"with_eve: {s['total_with_eve']:5d} | "
            f"correct: {s['correct']:5d} | "
            f"acc: {s['accuracy']:.2%}"
        )

import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(stats, total_questions):
    sources = SOURCE_ORDER
    x = np.arange(len(sources))
    width = 0.25

    col1 = []  # evidence / total
    col2 = []  # correct / evidence
    col3 = []  # correct / total

    for src in sources:
        eve = stats[src]["total_with_eve"]
        corr = stats[src]["correct"]

        col1.append(eve / total_questions if total_questions > 0 else 0)
        col2.append(corr / eve if eve > 0 else 0)
        col3.append(corr / total_questions if total_questions > 0 else 0)

    plt.figure(figsize=(10, 5))

    plt.bar(x - width, col1, width, label="Evidence / Total")
    plt.bar(x,         col2, width, label="Correct / Evidence")
    plt.bar(x + width, col3, width, label="Correct / Total")

    plt.xticks(x, sources, rotation=30, ha="right")
    plt.ylabel("Percentage")
    plt.ylim(0, 1)
    plt.title("Performance per Source")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    json_folder = "medhop_results/all_ans"
    stats, total_questions = analyze_accuracy_ignore_no_eve(json_folder)
    print_report(stats, total_questions)
    plot_histogram(stats, total_questions)

if __name__ == "__main__":
    main()
