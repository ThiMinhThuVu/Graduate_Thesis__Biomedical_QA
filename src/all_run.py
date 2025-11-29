from src.retrieval import *
from src.generate_answer import gen_ans_multiple_choice, generate_combine_parallel_answers
from src.decomposition import *
from main import generate_answer_for_subq

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
        retrieval_information, status, results_gg_search ,results_wiki_search = perform_retrieval(sample, search_kw, step, paths, settings, R7_info=R7_info)
        
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
        

def generate_ans_from_list_answer(question, list_answers, dataset=None):
    check_list_ans = set()

    long_ans_context = ""
    cnt = 0
    for item in list_answers:            
        ans = item["short_answer"]
        if ans.lower().strip() not in check_list_ans:
            label = chr(ord('A') + cnt)
            long_ans_context += f"{label}. {ans}\nreason: {item['long_answer']}\n\n"
            check_list_ans.add(ans.lower().strip())
            cnt += 1

    if len(list(check_list_ans)) == 1:
        return None
    
    try:
        result_json = gen_ans_multiple_choice(question, long_ans_context, dataset=dataset)
        
        if len(result_json) > 0 and 'A' <= result_json[0].upper() <= 'Z':
            selected_index = ord(result_json[0].upper()) - ord('A')
            if 0 <= selected_index < len(list_answers):
                selected_answer = list_answers[selected_index]

                return {
                    "selected_answer": result_json,
                    "long_answer": selected_answer["long_answer"],
                    "short_answer": selected_answer["short_answer"]
                }
            else:
                print(f"Lỗi: Chỉ số lựa chọn '{result_json}' không hợp lệ.")
        else:
            print(f"Lỗi: Phản hồi API '{result_json}' không phải là lựa chọn hợp lệ.")
    except Exception as e:
        print(f"Lỗi chọn câu trả lời cuối cùng.")
    return None
            
def preprocess_sample_R12(sample, settings, is_seq=False, dataset=None, type_q=None):
    sample_id = str(sample['QIDX']) 
    paths = setup_paths(settings, sample_id)
    file_all_ans = paths['final_answer']
    
    if os.path.exists(file_all_ans):
        print(f"All answers file already exists: {file_all_ans}")
        return
    
    if dataset == "medhop" and sample_id in settings['chrunk_mapping']:
        original_question = settings['chrunk_mapping'][sample_id]
    else:
        original_question = sample['Question']
        
    ans_old_path = paths["ans_old"] 
    all_context = ""
    
    if os.path.exists(ans_old_path):
        dict_ans = load_json(ans_old_path)  #dict_ans của R11
        n_qi = len(dict_ans.keys()) - 3 # 3 key khác không phải qi
        list_qi = []
        for i in range(n_qi):
            list_qi.append(dict_ans[f"{i+1}"]['final_qi']["question"])
    else:
        return 
    
    for q_i, question in enumerate(list_qi, start=1):
        qi_number = f"/{q_i}"
        q_i = str(q_i)
        paths = setup_paths(settings, sample_id, qi_number)
        
        subq_output = dict_ans[q_i]
        
        for path_i, path in subq_output.items():
            if path_i == "final_qi":
                continue
            for step, item in path.items():                
                step = str(step)
                subquestion = item['question']
                search_kw_subq = item['search_keyword']
                ans_old = item["short_answer"]
                R12_output = []
                R12_output.append({
                    "type_context": 0,
                    "question": subquestion,
                    "long_answer": item["long_answer"],
                    "short_answer": ans_old,
                    "retrieval_information": item["retrieval_information"]
                })

                for idx, type_context in enumerate(['vec_embed', 'only_gg'], 1): #"all"
                    short_ans = ""
                    if type_context == "only_gg":
                        key_entities = search_kw_subq + " " + short_ans
                    else:
                        if item['key_entity'] == "" or item['key_entity'] == None: 
                            key_entities = short_ans
                        else:
                            key_entities = item['key_entity']
                            
                    print(f"\n--- Processing Step {step}: {subquestion} ---")        
                    
                    if type_context == "all":
                        retrieval_information = R12_output[1]["retrieval_information"] + R12_output[2]["retrieval_information"]
                    else:
                        retrieval_information = perform_retrieval_R12(sample, key_entities, step, paths, settings, type_context=type_context)

                    answer_generation = generate_answer_for_subq(subquestion, key_entities, retrieval_information, step, paths, settings, dataset=dataset, type_q=type_q)
                    short_ans = answer_generation["short_answer"]
                    long_ans = answer_generation["long_answer"]
                    print(f"Short Answer for Step {step}: {short_ans}")
                    
                    R12_output.append({
                        "type_context": idx,
                        "question": subquestion,
                        "long_answer": long_ans,
                        "short_answer": short_ans,
                        "retrieval_information": item["retrieval_information"]
                    })
                
                multiple_choice_ans = generate_ans_from_list_answer(subquestion, R12_output, dataset=dataset)  
                if multiple_choice_ans == None: 
                    multiple_choice_ans = {
                        "selected_answer": "A",
                        "short_answer": item['short_answer'],
                        "long_answer": item['long_answer']
                    }                      
                     
                short_ans = multiple_choice_ans["short_answer"]
                long_ans = multiple_choice_ans["long_answer"]             

                dict_ans[q_i][path_i][step] = {"multiple_choice": {
                    "final_answer": multiple_choice_ans,
                    "list_ans": R12_output
                }}               

                if short_ans.lower() == "insufficient information": # sang path tiếp theo 
                    break
                
            # Lấy output step cuối cùng của qi:
            dict_ans[q_i]['final_qi']["question"] = question
            dict_ans[q_i]['final_qi']["search_keyword"] = search_kw_subq
            dict_ans[q_i]['final_qi']["key_entity"] = key_entities
            dict_ans[q_i]['final_qi']["short_answer"] = short_ans
            dict_ans[q_i]['final_qi']["long_answer"] = long_ans

            all_context += f"{q_i}.{question} - {short_ans}, {long_ans}\n\n"
           
    if n_qi > 1:
        combine_parallel_ans = generate_combine_parallel_answers(original_question, all_context, dataset=dataset, model="o3-mini")
        dict_ans['final_answer'] = combine_parallel_ans
        save_json(dict_ans, file_all_ans)
        print(f"Final answer: {combine_parallel_ans['short_answer']}")
        print(f"\nSaved to {file_all_ans}")
    else:
        dict_ans['final_answer'] = {
            'long_answer': dict_ans["1"]['final_qi']['long_answer'],
            'short_answer': dict_ans["1"]['final_qi']['short_answer']
        }
        save_json(dict_ans, file_all_ans)
        print(f"Final answer: {dict_ans['final_answer']['short_answer']}")
        print(f"\nSaved to {file_all_ans}")
        