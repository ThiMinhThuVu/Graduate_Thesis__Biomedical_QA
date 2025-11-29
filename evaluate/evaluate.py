import sys 
import json
import re
import string
from collections import Counter

run = 'R11' #'R14'
type_name = 'compositional' #'comparison'

pred_file = f'evaluate/run/{run}/pred_{type_name}.json'
gold_file = f'evaluate/run/{run}/gold_{type_name}.json'
question_scores_path = f'evaluate/run/{run}/question_scores_{type_name}.json'
alias_file = 'data/wiki/id_aliases.json'

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    # print(f'num_same: {num_same}, p_len: {len(prediction_tokens)}, gt_len: {len(ground_truth_tokens)}')
    # print(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def eval_answer(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall

def update_answer(metrics, prediction, golds, question_scores, cur_id):
    max_em, max_f1, max_prec, max_recall = 0, 0, 0, 0

    for gold in golds:
        em, f1, prec, recall = eval_answer(prediction, gold)
        max_em = max(max_em, em)
        max_f1 = max(max_f1, f1)
        max_prec = max(max_prec, prec)
        max_recall = max(max_recall, recall)

    metrics['em'] += float(max_em)
    metrics['f1'] += max_f1
    metrics['prec'] += max_prec
    metrics['recall'] += max_recall

    # Lưu điểm số cho câu hỏi hiện tại
    question_scores.append({
        'id': cur_id,
        'em': float(max_em),
        'f1': max_f1,
        'prec': max_prec,
        'recall': max_recall
    })

def eval(prediction_file, gold_file, alias_file):
    aliases = {}

    with open(prediction_file, encoding='utf-8') as f:
        prediction = json.load(f)
    with open(gold_file, encoding='utf-8') as f:
        gold = json.load(f)
    with open(alias_file, encoding='utf-8') as f:
        for json_line in map(json.loads, f):
            aliases[json_line["Q_id"]] = {
                "aliases": set(json_line["aliases"] + json_line["demonyms"])
            }

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    question_scores = []  # Danh sách lưu điểm số của từng câu hỏi

    for dp in gold:
        cur_id = dp['_id']
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
        else:
            gold_answers = {dp['answer']}  # Gold span
            if dp['answer_id'] in aliases and aliases[dp['answer_id']]["aliases"]:
                gold_answers.update(aliases[dp['answer_id']]["aliases"])
            update_answer(metrics, prediction['answer'][cur_id], gold_answers, question_scores, cur_id)

    N = len(gold)
    for k in metrics.keys():
        metrics[k] =  round(metrics[k] / N * 100, 2)

    print(f'{type_name}: {json.dumps(metrics, indent=4)}')
    
    question_scores = {
        item["id"]: item for item in question_scores
    }

    with open(question_scores_path, 'w', encoding='utf-8') as file:
        json.dump(question_scores, file)

if __name__ == '__main__':
    print(pred_file)
    print(gold_file)
    eval(pred_file, gold_file, alias_file)