import pandas as pd
from collections import Counter

# ======================
# CONFIG
# ======================

NO_EVIDENCE = "insufficient information"
MIN_VOTES = 3   # yêu cầu mới: >= 3 nguồn đồng thuận

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
# RULE: YN STRICT MAJORITY
# ======================

def strict_majority_vote_yn(row):
    """
    YN rule (STRICT):
    - final = yes nếu >= 3 nguồn vote yes
    - final = no  nếu >= 3 nguồn vote no
    - ngược lại => None
    """

    votes = []

    for src in SOURCE_PRIORITY:
        ans = normalize(row.get(src, ""))
        if ans in {"yes", "no"}:
            votes.append(ans)

    # không đủ evidence
    if len(votes) < MIN_VOTES:
        return None, "insufficient_votes"

    counter = Counter(votes)

    yes_cnt = counter.get("yes", 0)
    no_cnt  = counter.get("no", 0)

    if yes_cnt >= MIN_VOTES:
        return "yes", f"strict:{yes_cnt}-{no_cnt}"

    if no_cnt >= MIN_VOTES:
        return "no", f"strict:{no_cnt}-{yes_cnt}"

    return None, "no_strict_majority"


# ======================
# PIPELINE
# ======================

def run(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    results = df.apply(strict_majority_vote_yn, axis=1)

    df["final_answer"] = results.apply(lambda x: x[0])
    df["final_source"] = results.apply(lambda x: x[1])

    # correctness
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

    print("=== YN STRICT MAJORITY (>=3) ===")
    print(f"Total     : {total}")
    print(f"Answered  : {answered}")
    print(f"Correct   : {correct}")
    print(f"Accuracy  : {correct / total:.2%}")


# ======================
# MAIN
# ======================

def main():
    input_csv = r"evaluate\acc\short_answer_per_source.csv"
    output_csv = r"evaluate\acc\yn_strict_majority_vote.csv"
    run(input_csv, output_csv)
    print(output_csv)

if __name__ == "__main__":
    main()
