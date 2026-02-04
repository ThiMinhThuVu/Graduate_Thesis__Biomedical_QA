import os
import sys
import json
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from postprocessing import postprocessing
#config
run = 'R4_'
# run = 'R1_'

def load_json(golden_path):
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    return golden_data

with open("data/metadata_medhop.json", 'r', encoding='utf-8') as f:
    cache_class = json.load(f)


def load_predictions(pred_dir, golden_ids):
    preds = {}
    for fname in os.listdir(pred_dir):
        if not fname.startswith(run) or not fname.endswith('.json'):
            continue
        # Extract id from filename: R1_step_3b_ans_<id>.json
        try:
            id_part = fname.split('_')[-1].replace('.json', '')
        except Exception:
            continue
        if id_part not in golden_ids:
            continue
        with open(os.path.join(pred_dir, fname), 'r', encoding='utf-8') as f:
            try:
                pred_data = json.load(f)
                question = pred_data.get('original_question', '')
                answer = pred_data.get('final_answer', {}).get('short_answer').lower()
                context = pred_data['1']['final_qi']['retrieval_information'].lower() # Hop cuối
                
                type_q = cache_class[id_part]["type_q"]
                target_ans = cache_class[id_part]['target_answer_type']
                
                preds[id_part] = {'ans': answer, 'question': question, 'context': context, 'type_q': type_q, 'target_ans': target_ans}
            except Exception:
                preds[id_part] = {'ans': None, 'question': None, 'context': '', 'type_q': type_q, 'target_ans': target_ans}
    return preds

def evaluate_context(golden, preds, output_csv):
    list_id_gol = []
    count_check = 0
    count_check_norm = 0
    total = 0
    cnt_gol_ctx = 0
    yn = 0
    yesno_correct = 0
    direct_true = 0
    parallel_true = 0
    seq_true = 0
    direct_false = 0
    parallel_false = 0
    seq_false = 0
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['qidx', 'original question', 'golden', 'short answer', 'check_ans', 'answer norm', 'check_ans_norm', 'check_golden_in_context'])
        for qid, g in golden.items():
            golden_ans = g.get('True', '')
            pred_ans = preds.get(qid, {}).get('ans', None)
            question = preds.get(qid, {}).get('question', None)
            context = preds.get(qid, {}).get('context', '')
            type_q = preds.get(qid, {}).get('type_q', '')
            target_ans = preds.get(qid, {}).get('target_ans', '')
            if target_ans == 'correctness':
                continue
           
            check = False
            if pred_ans is not None:
                check = pred_ans.strip().lower() in golden_ans
            ans_norm = postprocessing(pred_ans) if pred_ans is not None else ''
            check_ans_norm = ans_norm in golden_ans
            if check:
                count_check += 1
            if check_ans_norm:
                list_id_gol.append(qid)
                count_check_norm += 1
                if type_q == 'direct':
                    direct_true += 1
                elif type_q == 'parallel':
                    parallel_true += 1
                elif type_q == 'sequence':
                    seq_true += 1
            else:
                if type_q == 'direct':
                    direct_false += 1
                elif type_q == 'parallel':
                    parallel_false += 1
                elif type_q == 'sequence':
                    seq_false += 1
                    
                
            # Đếm số câu trả lời yes/no đúng
            if pred_ans is not None and pred_ans.strip().lower() in ['yes', 'no'] and check:
                yesno_correct += 1
                
            total += 1
            
            check_golden_in_context = False
            if pred_ans not in ['yes', 'no', "insufficient information"]:
                for gol in golden_ans:
                    if gol in context:
                        check_golden_in_context = True
                        cnt_gol_ctx += 1
                        break
            else:
                yn += 1
                check_golden_in_context = None
            writer.writerow([qid, question, golden_ans, pred_ans, check, ans_norm, check_ans_norm, check_golden_in_context])
    # print("Yesno: ", yn)
    # total = 695
    total = 509 #direct
    print("Golden in context: ", cnt_gol_ctx)
    print(f"WH true (raw): {count_check}/{total}")
    print("WH false: ", total-count_check)
    print(f"WH true (norm): {count_check_norm}/{total}")
    
    # print("FULL YN + HOPCUOI: ", yn + cnt_gol_ctx)
    # print(f"Correct yes/no: {yesno_correct}")
    print("direct_true: ", direct_true)
    print("direct_false: ", direct_false)
    print("seq_true: ", seq_true)
    print("seq_false: ", seq_false)
    print("parallel_true: ", parallel_true)
    print("parallel_false: ", parallel_false)
    return list_id_gol

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
    
def main():        
    data_type1 = load_json("data_target_ans_type_old.json")
    data_type2 = load_json("data\metadata_medhop.json")
    # golden_path = os.path.join('data', 'all_answer_golden.json')
    golden_path = os.path.join('data', 'all_answer_golden_61.json')
    # pred_dir = os.path.join('medhop_results', 'all_ans')
    # pred_dir = r"E:\analysis_output\2026\code - cu -29.1\wh\target_type new\row14\all_ans"
    # pred_dir = r"E:\analysis_output\2026\code - cu -29.1\wh\row14\medhop_results\all_ans"
    # pred_dir = r"E:\analysis_output\2026\code - cu -29.1\R23\medhop_results\all_ans"
    pred_dir = r"E:\analysis_output\2026\code - cu -29.1\wh\target_type new\target2\row15\all_ans"
    # pred_dir = r"E:\UETQuintet\medhop_important_results\all_ans"
    
    output_csv = f'evaluate/run/{run}medhop_eval_result.csv'
    # golden = load_json(golden_path)
    # preds = load_predictions(pred_dir, set(golden.keys()))
    # list_id_gol_2 = evaluate_context(golden, preds, output_csv)

    # pred_dir_1 = r"E:\analysis_output\2026\code - cu -29.1\wh\row14\medhop_results\all_ans"
    # preds_1 = load_predictions(pred_dir_1, set(golden.keys()))
    # list_id_gol_1 = evaluate_context(golden, preds_1, output_csv)

    # tmp = list(set(list_id_gol_1) - set(list_id_gol_2))
    # print(len(tmp))
    # print(tmp)
    # print(f"Evaluation complete. Results saved to {output_csv}")
        
    golden = load_json(golden_path)
    preds = load_predictions(pred_dir, set(golden.keys()))
    list_id_gol_2 = evaluate_context(golden, preds, output_csv)

    # pred_dir_1 = r"E:\analysis_output\2026\code - cu -29.1\wh\row15\all_ans"
    # preds_1 = load_predictions(pred_dir_1, set(golden.keys()))
    # list_id_gol_1 = evaluate_context(golden, preds_1, output_csv)


    # # In ra số lượng và danh sách các phần tử khác biệt hoàn toàn giữa hai list
    # diff = list(set(list_id_gol_1) ^ set(list_id_gol_2))
    # print(f"Số lượng phần tử khác biệt hoàn toàn giữa hai list: {len(diff)}")
    # print("Danh sách các phần tử khác biệt hoàn toàn:")
    # print(diff)

    # tmp = list(set(list_id_gol_1) - set(list_id_gol_2))
    # print(len(tmp))
    # print(tmp)
    # # print(f"Evaluation complete. Results saved to {output_csv}")

    # # Xuất file so sánh list_id_gol_1 và list_id_gol_2
    # compare_csv = f'evaluate/run/{run}medhop_eval_compare.csv'
    # with open(compare_csv, 'w', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['qidx', 'question', 'golden','target1', 'ans_1', 'norm_ans_1', 'check_norm_1', 'target2', 'ans_2', 'norm_ans_2', 'check_norm_2', 'target'])
    #     for qid in set(list_id_gol_1).union(set(list_id_gol_2)):
    #         if qid not in diff:
    #             continue
    #         question = preds_1.get(qid, {}).get('question', '')
    #         golden_ans = golden[qid]["True"]
    #         ans_1 = preds_1.get(qid, {}).get('ans', '')
    #         norm_ans_1 = postprocessing(ans_1) if ans_1 else ''
    #         check_norm_1 = norm_ans_1 in golden_ans
    #         target1 = data_type1[qid]['target_answer_type']
    #         ans_2 = preds.get(qid, {}).get('ans', '')
    #         norm_ans_2 = postprocessing(ans_2) if ans_2 else ''
    #         check_norm_2 = norm_ans_2 in golden_ans
    #         target2 = data_type2[qid]['target_answer_type']
    #         if check_norm_1 and check_norm_2 is False and target1 != target2:
    #             target = target1
    #             cache_class[qid]['target_answer_type'] = target
    #             writer.writerow([qid, question, golden_ans, target1, ans_1, norm_ans_1, check_norm_1, target2, ans_2, norm_ans_2, check_norm_2, target])
    # print(f"Comparison CSV saved to {compare_csv}")
    # save_json(cache_class, "data/metadata_medhop.json")

if __name__ == '__main__':
    main()
