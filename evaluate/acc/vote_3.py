import pandas as pd
from collections import Counter, defaultdict
from postprocessing import postprocessing

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


def valid(ans):
    return ans and ans != NO_EVIDENCE


# ======================
# RULE: ANY-2 AGREEMENT
# ======================
# def any_two_agreement(row):
#     """
#     Rule:
#     - Lấy answer nếu có >= 2 nguồn bất kỳ giống nhau
#     - Nếu không có => None
#     """

#     answers = {}

#     # collect valid answers
#     for src in SOURCE_PRIORITY:
#         ans = normalize(row.get(src, ""))
#         if valid(ans):
#             answers[src] = ans

#     # no evidence at all
#     if not answers:
#         return None, "no_evidence"

#     # count answers
#     counter = Counter(answers.values())

#     # answers with >= 2 votes
#     agreed = [ans for ans, cnt in counter.items() if cnt >= 3]

#     # ❌ KHÔNG có agreement
#     if not agreed:
#         return None, "no_agreement"

#     # ✅ có agreement → chọn theo source priority
#     for src in SOURCE_PRIORITY:
#         if src in answers and answers[src] in agreed:
#             return answers[src], f"any2:{src}"

#     # theoretically unreachable
#     return None, "no_agreement"


def canonical_source(src):
    """
    wiki và crawl_wiki được coi là cùng 1 nguồn
    """
    if src in {"wiki", "crawl_wiki"}:
        return "wiki"
    return src

# # loại trừ wiki=crawl_wiki
# def any_two_agreement(row):
#     """
#     WH rule:
#     - wiki == crawl_wiki (coi là cùng nguồn)
#     - Lấy answer nếu có >= 2 nguồn ĐỘC LẬP giống nhau
#     - Ngược lại => None
#     """

#     answers = {}

#     # collect valid answers (sau khi canonicalize source)
#     for src in SOURCE_PRIORITY:
#         ans = normalize(row.get(src, ""))
#         if valid(ans):
#             canon_src = canonical_source(src)
#             answers.setdefault(ans, set()).add(canon_src)

#     # no evidence
#     if not answers:
#         return None, "no_evidence"

#     valid_answers = []
#     # list_norm = []

#     for ans, src_set in answers.items():
#         # cần ít nhất 2 nguồn độc lập
#         if len(src_set) >= 2:
#             valid_answers.append(ans)
#             # ans_norm = postprocessing(ans) 
#             # list_norm.append(ans_norm)

#     if not valid_answers:
#         return None, "no_valid_agreement"

#     # chọn theo SOURCE_PRIORITY (sau canonical)
#     for src in SOURCE_PRIORITY:
#         canon_src = canonical_source(src)
#         for ans in valid_answers:
#             if canon_src in answers.get(ans, set()):
#                 return ans, f"any2:{canon_src}"

#     return None, "no_valid_agreement"


def any_two_agreement(row):
    """
    Any-2 Agreement Rule
    """

    answers = {}

    # collect valid answers
    for src in SOURCE_PRIORITY:
        ans = normalize(row.get(src, ""))
        if valid(ans):
            answers[src] = ans

    # no evidence at all
    if not answers:
        return NO_EVIDENCE, "no_evidence"

    # count answers
    counter = Counter(answers.values())

    # answers with >=2 votes
    agreed = [ans for ans, cnt in counter.items() if cnt >= 2]

    # Case 1: found agreement
    if agreed:
        # tie-break by source priority
        for src in SOURCE_PRIORITY:
            if src in answers and answers[src] in agreed:
                return answers[src], f"any2:{src}"

    # Case 2: only single answers
    if len(answers) == 1:
        src, ans = next(iter(answers.items()))
        return ans, f"single:{src}"

    # fallback (should rarely happen)
    src, ans = next(iter(answers.items()))
    return ans, f"fallback:{src}"


# ======================
# PIPELINE
# ======================

# def run(input_csv, output_csv):
#     df = pd.read_csv(input_csv)

#     results = df.apply(any_two_agreement, axis=1)

#     df["final_answer"] = results.apply(lambda x: x[0])
#     df["final_source"] = results.apply(lambda x: x[1])

#     df["final_correct"] = (
#         df["final_answer"].fillna("").apply(normalize)
#         == df["golden"].apply(normalize)
#     )


#     out = df[[
#         "QIDX",
#         "golden",
#         "final_answer",
#         "final_source",
#         "final_correct"
#     ]]

#     out.to_csv(output_csv, index=False)

#     total = len(out)
#     correct = out["final_correct"].sum()
#     # answered = (out["final_answer"].apply(normalize) != NO_EVIDENCE).sum()
#     answered = df["final_answer"].notna().sum()

#     print("=== RULE: ANY-2 AGREEMENT ===")
#     print(f"Total    : {total}")
#     print(f"Answered : {answered}")
#     print(f"Correct  : {correct}")
#     print(f"Accuracy : {correct / answered:.2%}")
    

def run(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    results = df.apply(any_two_agreement, axis=1)

    df["final_answer"] = results.apply(lambda x: x[0])
    df["final_source"] = results.apply(lambda x: x[1])

    # ✅ NEW: norm answer
    df["final_answer_norm"] = (
        df["final_answer"]
        .fillna("")
        .apply(postprocessing)
    )

    df["final_correct"] = (
        df["final_answer_norm"].apply(normalize)
        == df["golden"].apply(normalize)
    )

    out = df[[
        "QIDX",
        "golden",
        "final_answer",
        "final_answer_norm",   # ✅ thêm cột
        "final_source",
        "final_correct"
    ]]

    out.to_csv(output_csv, index=False)

    total = len(out)
    correct = out["final_correct"].sum()
    
    # answered = df["final_answer"].notna().sum()
    answered = (out["final_answer"].apply(normalize) != NO_EVIDENCE).sum()

    print("=== RULE: ANY-2 AGREEMENT ===")
    print(f"Total    : {total}")
    print(f"Answered : {answered}")
    print(f"Correct  : {correct}")
    print(f"Accuracy : {correct / answered:.2%}")


def main():
    input_csv = r"evaluate\acc\short_answer_per_source.csv"
    output_csv = r"evaluate\acc\any2_agreement.csv"
    run(input_csv, output_csv)
    print(output_csv)

if __name__ == "__main__":
    main()
