import pandas as pd
from collections import Counter
from postprocessing import postprocessing

# ==============================
# CONFIG
# ==============================

NO_EVIDENCE = "insufficient information"

SOURCE_PRIORITY = [
    "crawl_sparql_wikidata",
    "google",
    "crawl_ner_wiki",
    "wiki",
    "crawl_wiki",    
]

# ==============================
# CORE LOGIC
# ==============================

def normalize(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def majority_vote_with_priority(row):
    """
    Majority Voting + Priority tie-break
    Return:
        final_answer, final_source, count
    """

    answers = {}

    # 1. Collect valid answers
    for src in SOURCE_PRIORITY:
        ans = normalize(row.get(src, ""))
        if ans and ans != NO_EVIDENCE:
            answers[src] = ans

    # 2. No evidence at all
    if not answers:
        return NO_EVIDENCE, "none", 0

    # 3. Majority voting
    counter = Counter(answers.values())
    max_freq = max(counter.values())

    top_answers = [
        ans for ans, cnt in counter.items()
        if cnt == max_freq
    ]

    # 4. Unique winner
    if len(top_answers) == 1:
        final_ans = top_answers[0]
        for src in SOURCE_PRIORITY:
            if answers.get(src) == final_ans:
                return final_ans, src, max_freq

    # 5. Tie-break by priority
    for src in SOURCE_PRIORITY:
        if src in answers and answers[src] in top_answers:
            return answers[src], src, max_freq

    # Fallback
    final_ans = top_answers[0]
    for src, ans in answers.items():
        if ans == final_ans:
            return final_ans, src, max_freq

    return NO_EVIDENCE, "none", 0


# ==============================
# PIPELINE
# ==============================
# ==============================
# PIPELINE
# ==============================
def run_majority_voting(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    # Apply voting rule
    results = df.apply(majority_vote_with_priority, axis=1)

    # ✅ unpack results
    df["final_answer"] = results.apply(lambda x: x[0])
    df["final_source"] = results.apply(lambda x: x[1])
    # nếu rule có count thì bật dòng dưới
    # df["vote_count"] = results.apply(lambda x: x[2])

    # ✅ NEW: normalized final answer
    df["final_answer_norm"] = (
        df["final_answer"]
        .fillna("")
        .apply(postprocessing)
    )

    # Compare with golden
    df["final_correct"] = (
        df["final_answer_norm"].apply(normalize)
        == df["golden"].apply(normalize)
    )

    # Export required columns
    out_df = df[[
        "QIDX",
        "golden",
        "final_answer",
        "final_answer_norm",
        "final_source",
        "final_correct"
    ]]

    out_df.to_csv(output_csv, index=False)

    # Report
    total = len(out_df)
    correct = out_df["final_correct"].sum()
    answered = out_df["final_answer"].notna().sum()
    acc = correct / answered if answered > 0 else 0
    none_count = out_df["final_answer"].isna().sum()

    print("=== MAJORITY VOTING RESULT ===")
    print(f"Total    : {total}")
    print(f"Answered : {answered}")
    print(f"Correct  : {correct}")
    print(f"Accuracy : {acc:.2%}")
    print(f"None     : {none_count}")


def main():
    input_csv = r"evaluate\acc\short_answer_per_source.csv"
    output_csv = r"evaluate\acc\vote_1.csv"
    run_majority_voting(input_csv, output_csv)


if __name__ == "__main__":
    main()
