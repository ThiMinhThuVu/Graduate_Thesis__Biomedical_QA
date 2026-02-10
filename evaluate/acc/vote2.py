import pandas as pd
from postprocessing import postprocessing

# ==============================
# CONFIG
# ==============================

NO_EVIDENCE = "insufficient information"

PRIMARY = "google"
SECONDARY = "wiki"
CRAWL = "crawl_wiki"

SOURCES = [
    "google",
    "wiki",
    "crawl_wiki",
    "crawl_ner_wiki",
    "crawl_sparql_wikidata"
]

# ==============================
# UTILS
# ==============================

def normalize(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def valid(ans):
    return ans and ans != NO_EVIDENCE


# ==============================
# TIERED CONFIDENCE RULE
# ==============================

def tiered_confidence(row):
    answers = {
        src: normalize(row.get(src, ""))
        for src in SOURCES
        if valid(normalize(row.get(src, "")))
    }

    # No evidence at all
    if not answers:
        return NO_EVIDENCE, "none"

    g = answers.get("google")
    w = answers.get("wiki")
    c = answers.get("crawl_wiki")

    # ==========================
    # RULE 1: Wiki == Google
    # ==========================
    if g and w and g == w:
        return g, "google+wiki"

    # ==========================
    # RULE 2: Wiki == Crawl_wiki != Google
    # ==========================
    if w and c and w == c and w != g:
        return w, "wiki+crawl_wiki"

    # ==========================
    # RULE 4: Single source only
    # ==========================
    if len(answers) == 1:
        src, ans = next(iter(answers.items()))
        if src in {"google", "wiki"}:
            return ans, src
        else:
            return NO_EVIDENCE, "none"

    # ==========================
    # RULE 3: All different
    # ==========================
    if g:
        return g, "google_fallback"

    # fallback (should rarely happen)
    src, ans = next(iter(answers.items()))
    return ans, src


# ==============================
# PIPELINE
# ==============================

# def run_tiered_confidence(input_csv, output_csv):
#     df = pd.read_csv(input_csv)

#     results = df.apply(tiered_confidence, axis=1)
#     df["final_answer"] = results.apply(lambda x: x[0])
#     df["final_source"] = results.apply(lambda x: x[1])

#     df["final_correct"] = (
#         df["final_answer"].apply(normalize)
#         == df["golden"].apply(normalize)
#     )

#     out_df = df[
#         ["QIDX", "golden", "final_answer", "final_source", "final_correct"]
#     ]

#     out_df.to_csv(output_csv, index=False)

#     total = len(out_df)
#     correct = out_df["final_correct"].sum()
#     answered = (out_df["final_answer"].apply(normalize) != NO_EVIDENCE).sum()

#     print("=== TIERED CONFIDENCE RESULT ===")
#     print(f"Total    : {total}")
#     print(f"Correct  : {correct}")
#     print(f"Accuracy : {correct / total:.2%}")
#     print(f"Answered : {answered}")

def run_tiered_confidence(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Apply rule
    results = df.apply(tiered_confidence, axis=1)
    df["final_answer"] = results.apply(lambda x: x[0])
    df["final_source"] = results.apply(lambda x: x[1])

    # ✅ NEW: normalized final answer
    df["final_answer_norm"] = (
        df["final_answer"]
        .fillna("")
        .apply(postprocessing)
    )

    # Compare with golden (using normalized answer)
    df["final_correct"] = (
        df["final_answer_norm"].apply(normalize)
        == df["golden"].apply(normalize)
    )

    out_df = df[
        [
            "QIDX",
            "golden",
            "final_answer",
            "final_answer_norm",   # ✅ thêm cột
            "final_source",
            "final_correct"
        ]
    ]

    out_df.to_csv(output_csv, index=False)

    total = len(out_df)
    correct = out_df["final_correct"].sum()
    answered = (out_df["final_answer"].apply(normalize) != NO_EVIDENCE).sum()

    print("=== TIERED CONFIDENCE RESULT ===")
    print(f"Total    : {total}")
    print(f"Answered : {answered}")
    print(f"Correct  : {correct}")
    print(f"Accuracy : {correct / answered:.2%}")

def main():
    input_csv = r"evaluate\acc\short_answer_per_source.csv"
    output_csv = r"evaluate\acc\tiered_confidence_2.csv"
    run_tiered_confidence(input_csv, output_csv)


if __name__ == "__main__":
    main()
