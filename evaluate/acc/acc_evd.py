import os
import json

SOURCE_ORDER = [
    "google",
    "wiki",
    "crawl_wiki",
    "crawl_ner_wiki",
    "crawl_sparql_wikidata"
]

with open('data/meta_entity_YN.json', "r", encoding="utf-8") as f:
    metadata = json.load(f)
    
# with open('data/meta_entity_WH.json', "r", encoding="utf-8") as f:
#     metadata = json.load(f)
    
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
        
        if qidx not in metadata:
            continue

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

    raw_col1 = []  # eve
    raw_col2 = []  # correct
    raw_col3 = []  # correct

    for src in sources:
        eve = stats[src]["total_with_eve"]
        corr = stats[src]["correct"]

        col1.append(eve / total_questions if total_questions > 0 else 0)
        col2.append(corr / eve if eve > 0 else 0)
        col3.append(corr / total_questions if total_questions > 0 else 0)

        raw_col1.append(eve)
        raw_col2.append(corr)
        raw_col3.append(corr)

    plt.figure(figsize=(11, 5))

    bars1 = plt.bar(x - width, col1, width, label="Evidence / Total")
    bars2 = plt.bar(x,         col2, width, label="Correct / Evidence")
    bars3 = plt.bar(x + width, col3, width, label="Correct / Total")

    plt.xticks(x, sources, rotation=30, ha="right")
    plt.ylabel("Percentage")
    plt.ylim(0, 1)
    plt.title("Performance per Source")
    plt.legend()

    # ===== annotate =====
    def annotate(bars, values, denoms):
        for bar, v, d in zip(bars, values, denoms):
            height = bar.get_height()
            pct = height * 100

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{v}/{d}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8
            )

    annotate(bars1, raw_col1, [total_questions]*len(sources))
    annotate(bars2, raw_col2, raw_col1)
    annotate(bars3, raw_col3, [total_questions]*len(sources))

    plt.tight_layout()
    plt.show()


import csv

def export_short_answer_csv(json_folder, output_csv="short_answer_analysis.csv"):
    rows = []

    for fname in os.listdir(json_folder):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(json_folder, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        qidx = data["QIDX"]
        
        if qidx not in metadata:
            continue

        golden_list = dict_golden[qidx]["True"]
        golden_norm = [g.lower() for g in golden_list]

        source_outputs = data['1']["0"]["0"].get("source_output", {})

        for src, result in source_outputs.items():
            src = normalize_source(src)
            if src not in SOURCE_ORDER:
                continue

            short_answer = result.get("short_answer", "").strip()
            short_norm = short_answer.lower()

            has_evidence = short_norm != "insufficient information"

            correct = False
            if has_evidence:
                for g in golden_norm:
                    if short_norm == g:
                        correct = True
                        break

            rows.append({
                "QIDX": qidx,
                "source": src,
                "short_answer": short_answer,
                "golden": " | ".join(golden_list),
                "has_evidence": has_evidence,
                "correct": correct
            })

    # ghi CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "QIDX",
                "source",
                "short_answer",
                "golden",
                "has_evidence",
                "correct"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] CSV exported to: {output_csv}")

def main():
    json_folder = "medhop_results/all_ans"

    stats, total_questions = analyze_accuracy_ignore_no_eve(json_folder)
    print_report(stats, total_questions)
    plot_histogram(stats, total_questions)

    export_short_answer_csv(
        json_folder,
        output_csv="evaluate/acc/short_answer_per_source.csv"
    )

if __name__ == "__main__":
    main()
