import os
import json
from collections import defaultdict, Counter, OrderedDict
import matplotlib.pyplot as plt


# Thứ tự source theo pipeline settings
SOURCE_ORDER = [
    "google",
    "wiki",
    "crawl_wiki",
    "crawl_ner_wiki",
    "crawl_sparql_wikidata"
]


def normalize_source(source):
    """
    Chuẩn hoá tên source nếu cần
    """
    if not source:
        return "unknown"
    return source.strip().lower()


def analyze_insufficient_answers(json_folder):
    """
    Phân tích các candidate có short_answer = 'Insufficient information'
    """

    source_counter = Counter()
    detail_by_question = defaultdict(list)

    total_samples = 0
    total_candidates = 0
    total_insufficient = 0

    for fname in os.listdir(json_folder):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(json_folder, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Cannot load {fname}: {e}")
            continue

        total_samples += 1
        qid = data.get("QIDX", fname)

        for k, v in data.items():
            if not k.isdigit():
                continue

            final_qi = v.get("final_qi", {})
            candidates = final_qi.get("candidate", [])

            for cand in candidates:
                total_candidates += 1
                source = normalize_source(cand.get("source", "unknown"))

                if cand.get("short_answer").lower() == "insufficient information":
                    total_insufficient += 1
                    source_counter[source] += 1
                    detail_by_question[qid].append(source)

    return (
        source_counter,
        detail_by_question,
        total_samples,
        total_candidates,
        total_insufficient
    )


def reorder_sources(counter):
    ordered = OrderedDict()

    for src in SOURCE_ORDER:
        ordered[src] = counter.get(src, 0)

    for src, cnt in counter.items():
        if src not in ordered:
            ordered[src] = cnt

    return ordered



def save_stats(source_counter, detail_by_question, meta_info, output_path):
    output = {
        "meta": meta_info,
        "total_insufficient_by_source": dict(source_counter),
        "detail_by_question": dict(detail_by_question)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

def plot_histogram(source_counter, total_insufficient):
    if not source_counter:
        print("[INFO] No insufficient information cases found.")
        return

    sources = list(source_counter.keys())
    counts = list(source_counter.values())

    plt.figure(figsize=(9, 5))
    bars = plt.bar(sources, counts)

    plt.xlabel("Source")
    plt.ylabel("Count")
    plt.title(f"Insufficient Information by Source (Total = {total_insufficient})")
    plt.xticks(rotation=30)
    
    plt.ylim(0, 800)


    # Hiển thị số lượng + % trên mỗi cột
    for bar, cnt in zip(bars, counts):
        height = bar.get_height()
        ratio = cnt / total_insufficient if total_insufficient > 0 else 0.0

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(cnt)}\n({ratio:.1%})",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


def main():
    json_folder = "medhop_results_no_mcq_TFIDF300/all_ans"
    output_json = "evaluate/acc/insufficient_stats.json"

    (
        source_counter,
        detail_by_question,
        total_samples,
        total_candidates,
        total_insufficient
    ) = analyze_insufficient_answers(json_folder)

    # Sắp xếp source theo pipeline
    source_counter = reorder_sources(source_counter)

    print("\n=== SUMMARY ===")
    print(f"Total samples (questions): {total_samples}")
    print(f"Total candidates: {total_candidates}")
    print(f"Total insufficient answers: {total_insufficient}\n")

    print("=== Insufficient Information by Source ===")
    for src, cnt in source_counter.items():
        ratio = cnt / total_insufficient if total_insufficient > 0 else 0
        print(f"{src:25s}: {cnt:5d}  ({ratio:.2%})")

    meta_info = {
        "total_samples": total_samples,
        "total_candidates": total_candidates,
        "total_insufficient": total_insufficient
    }

    save_stats(source_counter, detail_by_question, meta_info, output_json)
    plot_histogram(source_counter, total_insufficient)


if __name__ == "__main__":
    main()
