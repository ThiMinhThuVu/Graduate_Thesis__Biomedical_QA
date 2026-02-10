import pandas as pd
from collections import Counter

# ======================
# CONFIG
# ======================

NO_EVIDENCE = "insufficient information"

SOURCE_PRIORITY = [
    "google",
    "wiki",
    "crawl_wiki",
    "crawl_ner_wiki",
    "crawl_sparql_wikidata"
]

# ======================
# UTILS
# ======================

def normalize(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


# ======================
# RULE: YN MAJORITY VOTE
# ======================

def majority_vote_yn(row):
    """
    YN rule:
    - Collect yes / no from all sources
    - Majority wins
    - Tie or no evidence -> None
    """

    votes = []

    for src in SOURCE_PRIORITY:
        ans = normalize(row.get(src, ""))
        if ans in {"yes", "no"}:
            votes.append(ans)

    # no evidence
    if not votes:
        return None, "no_evidence"

    counter = Counter(votes)

    yes_cnt = counter.get("yes", 0)
    no_cnt  = counter.get("no", 0)

    # majority wins
    if yes_cnt > no_cnt:
        return "yes", f"majority:{yes_cnt}-{no_cnt}"

    if no_cnt > yes_cnt:
        return "no", f"majority:{no_cnt}-{yes_cnt}"

    # tie
    return None, "tie"


# ======================
# PIPELINE
# ======================

def run(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    results = df.apply(majority_vote_yn, axis=1)

    df["final_answer"] = results.apply(lambda x: x[0])
    df["final_source"] = results.apply(lambda x: x[1])

    # correctness (YN only)
    df["final_correct"] = (
        df["final_answer"].fillna("")
        == df["golden"].apply(normalize)
    )

    out = df[[
        "QIDX",
        "golden",
        "final_answer",
        "final_source",
        "final_correct"
    ]]

    out.to_csv(output_csv, index=False)

    # stats
    total = len(out)
    answered = out["final_answer"].notna().sum()
    correct = out["final_correct"].sum()

    print("=== YN MAJORITY VOTE ===")
    print(f"Total     : {total}")
    print(f"Answered  : {answered}")
    print(f"Correct   : {correct}")
    print(f"Accuracy  : {correct / answered:.2%}")


# ======================
# MAIN
# ======================

def main():
    input_csv = r"evaluate\acc\short_answer_per_source.csv"
    output_csv = r"evaluate\acc\yn_majority_vote.csv"
    run(input_csv, output_csv)


if __name__ == "__main__":
    main()
