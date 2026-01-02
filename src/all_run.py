from src.retrieval import *
from src.generate_answer import gen_ans_multiple_choice, generate_combine_parallel_answers
from src.decomposition import *
from main_hop2_wikidata_dev import generate_answer_for_subq

def process_answer_with_snippet_removal(sample, search_kw, step, paths, settings, R4_output_for_step, dataset=None):
    """
    Xử lý quá trình tạo câu trả lời với cơ chế xóa snippet và chọn câu trả lời cuối cùng.
    """
    R7_output = []    
    ans_old = R4_output_for_step["short_answer"] 
    question = R4_output_for_step['question']
    R7_output.append({
        "del_snippet": 0,
        "question": question,
        "long_answer": R4_output_for_step["long_answer"],
        "short_answer": ans_old
    })

    status = True
    idx_del = 1
    status_cnt = 0
    
    if settings['get_gg']:
        results_gg_search = retrieval_google(search_kw, step, paths['gg_search'])            
        
    if settings['get_wiki']:
        results_wiki_search = retrieval_wiki(search_kw, step, paths['wiki_search'])
        
    # Xóa snippet
    while status:
        status_cnt += 1
        if status_cnt >= 10:
            status = False        
        R7_info = {
            'ans_old': ans_old,
            'results_gg_search': results_gg_search,
            'results_wiki_search': results_wiki_search
        }
        key_entity = ""
        retrieval_information, status, results_gg_search ,results_wiki_search = perform_retrieval(sample, key_entity, search_kw, step, paths, settings, R7_info=R7_info)
        
        if status: # Nếu có sự thay đổi snippet hoặc tìm thấy snippet mới để xóa
            answer_generation = generate_answer_for_subq(question, search_kw, retrieval_information, step, paths, settings, dataset=dataset, type_q=sample['type'])
            if ans_old != answer_generation["short_answer"]:
                ans_old = answer_generation["short_answer"]
                R7_output.append({
                    "del_snippet": idx_del,
                    "question": question,
                    "long_answer": answer_generation["long_answer"],
                    "short_answer": ans_old
                })
            else:
                status = False 
        idx_del += 1

    check_list_ans = set()
    error = False
    final_short_answer = ans_old 

    if len(R7_output) > 1:
        long_ans_context = ""
        for i, item in enumerate(R7_output):
            label = chr(ord('A') + i)
            ans = item["short_answer"]
            if ans.lower().strip() not in check_list_ans:
                long_ans_context += f"{label}. {ans}\nreason: {item['long_answer']}\n\n"
                check_list_ans.add(ans.lower().strip())
        
        try:
            result_json = gen_ans_multiple_choice(question, long_ans_context, dataset=dataset)
           
            if len(result_json) > 0 and 'A' <= result_json[0].upper() <= 'Z':
                selected_index = ord(result_json[0].upper()) - ord('A')
                if 0 <= selected_index < len(R7_output):
                    selected_answer = R7_output[selected_index]
                    final_short_answer = selected_answer["short_answer"]

                    R7_output.append({
                        "selected_answer": result_json,
                        "long_answer": selected_answer["long_answer"],
                        "short_answer": selected_answer["short_answer"]
                    })
                else:
                    print(f"Lỗi: Chỉ số lựa chọn '{result_json}' không hợp lệ cho bước {step}.")
                    error = True
            else:
                print(f"Lỗi: Phản hồi API '{result_json}' không phải là lựa chọn hợp lệ cho bước {step}.")
                error = True
        except Exception as e:
            print(f"Lỗi chọn câu trả lời cuối cùng bước {step}: {e}")
            error = True
            
    return final_short_answer, R7_output, error

def preprocess_sample_R7(sample, settings, dataset=None):
    sample_id = str(sample['QIDX']) 
    paths = setup_paths(settings, sample_id)
    file_all_ans = paths['final_answer']
    
    if os.path.exists(file_all_ans):
        print(f"All answers file already exists: {file_all_ans}")
        return
    
    ans_old_path = paths["ans_old"] 

    if os.path.exists(ans_old_path):
        dict_ans = load_json(ans_old_path)  #dict_ans của R4
        n_qi = len(dict_ans.keys()) - 3 # 3 key khác không phải qi
        list_qi = []
        for i in range(n_qi):
            list_qi.append(dict_ans[f"{i+1}"]['final_qi']["question"])
    else:
        return 
    
    for q_i, question in enumerate(list_qi, start=1):
        qi_number = f"/{q_i}"
        paths = setup_paths(settings, sample_id, qi_number)
        
        subq_output = dict_ans[str(q_i)]
        short_ans = ""
        for path_i, path in subq_output.items():
            if path_i == "final_qi":
                continue
            for step, item in path.items():                
                step = str(step)
                subquestion = item['question']
                search_kw_subq = item['search_keyword']
                
                if settings['search_key_entity'] == "True": 
                    if item['key_entity'] == "" or item['key_entity'] == None: 
                        key_entities = short_ans
                    else:
                        key_entities = item['key_entity']        
                else:
                    key_entities = search_kw_subq + " " + short_ans
                    
                R4_output_for_step = item
                short_ans, R7_output_for_step, error = process_answer_with_snippet_removal(sample, key_entities, step, paths, settings, R4_output_for_step, dataset=dataset)

                # update câu trả lời ngắn cho bước tiếp theo
                if not(error):
                    dict_ans[str(q_i)][str(path_i)][step]['multiple_choice'] = R7_output_for_step  
                    print(f"Short Answer for Step {step}:\n{short_ans}")
                else:
                    print(f"Step 7b error because GPT response is not expected")
                    
    save_json(dict_ans, file_all_ans)
    print(f"\nFinal answer saved to {file_all_ans}")
    print("=" * 50, "\nOverall Short Answer:\n", short_ans, "\n" + "=" * 50)
        