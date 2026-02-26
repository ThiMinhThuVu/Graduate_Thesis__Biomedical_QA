import json
import re
import string
import copy

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def get_longest_capitalized_substrings(s: str):
    question_words = {"what", "who", "when", "where", "why", "how", "which"}

    words = s.split()
    result = []

    n = len(words)
    i = 0

    while i < n:
        if not words[i][0].isupper():
            i += 1
            continue

        j = i + 1
        while j < n:
            if words[j][0].isupper():
                j += 1
            elif j + 1 < n and words[j + 1][0].isupper():
                j += 1
            else:
                break

        phrase = " ".join(words[i:j])
        phrase = phrase.strip(string.punctuation)
        phrase = phrase.strip()

        if len(phrase) >= 3 and len(phrase.split(' ')) != 1:
            first_word = phrase.split()[0].lower()
            if first_word not in question_words:
                result.append(phrase)

        i = j

    return result


def extract_quoted_strings(text: str):
    pattern = r"'([^']*)'|\"([^\"]*)\""
    matches = re.findall(pattern, text)
    return [m[0] if m[0] else m[1] for m in matches]

# data1 = load_json(r'data\get_entity_from_question_spacy_parallel_seq.json')
# data2 = load_json(r'data\entity_extraction_results_LLM.json')

data1 = load_json(r'data\ner\SEQ\spacy_entity_9000q_1.json')
data2 = load_json(r'data\ner\LLM\ner_llm_9000_seq.json')
# data3 = load_json(r'data\ner\LLM\LLM_entity_9000q_2.json')
# data4 = load_json(r'data\ner\LLM\LLM_entity_9000q_3.json')
# data2.update(data3)
# data2.update(data4)


data_search = []
data_search_lower = []
type = []
result_dict = {}

OUTPUT_TXT = "output_entities.txt"

merge_dict = copy.copy(data1)
for qidx in data1:
    question = data1[qidx]["question"]

    arr = get_longest_capitalized_substrings(question)
    # if len(arr) > 0:
    #     print('hello')

    for tmp in get_longest_capitalized_substrings(question):
        data1[qidx]["extracted_entities"]["detail"][tmp] = {
            "label": "new_1",
            "bio_type": None,
            "source": ['custom']
        }
    
    # if len(get_longest_capitalized_substrings(question)) > 0:
    #     print(get_longest_capitalized_substrings(question))

    for tmp in extract_quoted_strings(question):
        data1[qidx]["extracted_entities"]["detail"][tmp] = {
            "label": "new_2",
            "bio_type": None,
            "source": ['custom']
        }


    extracted_entities = []
    for qid, tmp in data2.items():
        # print(qidx)
        # print(tmp)
        if str(qidx) == str(tmp['id']):
            # extracted_entities = json.loads(tmp['extracted_entities'])
            extracted_entities = tmp['entities']
            for extract_entity in extracted_entities:
                if len(extract_entity.split(' ')) < 2 and extract_entity not in data1[qidx]["extracted_entities"]["detail"]:
                    continue
                if extract_entity in data1[qidx]["extracted_entities"]["detail"]:
                    continue

                data1[qidx]["extracted_entities"]["detail"][extract_entity] = {
                    "label": "gpt",
                    "bio_type": None,
                    "source": ['gpt']
                }
            break
    

    sorted_dict = dict(
        sorted(
            data1[qidx]["extracted_entities"]["detail"].items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
    )

    printed_array = []
    entities_full = []

    cnt = 0
    cnt_gpt = 0
    merge_dict[qidx]["extracted_entities"]["detail"] = {}
    
    for entity, info in sorted_dict.items():
        if cnt == 5 or cnt_gpt == 5:
            # print(qidx)
            break

        if info['label'] in [
            "ORGAN",
            "DEVELOPING_ANATOMICAL_STRUCTURE",
            "CELL_LINE",
            "IMMATERIAL_ANATOMICAL_ENTITY"
        ]:
            continue

        if len(entity) <= 3 and info['label'] == 'new_1':
            continue
        elif len(entity) <= 2:
            continue

        if len(entity.split(' ')) == 1 and info['bio_type'] == 'non-bio':
            continue

        if len(entity.split(' ')) == 1 and info['label'] == 'ORGANISM':
            continue 

        if len(entity.split(' ')) == 1 and info['label'] == 'TISSUE':
            continue

        if entity.upper() in ["CELL", "CELLS", "CELLULAR", "HOST CELL", "RNA", "DNA", "DISEASE", "DISEASES", "TOOTH","FEVER","ENZYME","OVERDOSE","FS’","DEATH","IRON","EYES","CANCERS","ENZYMES","PAIN","CHROMOSOMES","CHROMOSOME","DEATHS","FISHY","CANCER","FUNGUS","DELETIONS"]:
            continue

        entity_in_entity = False
        for tmp in printed_array:
            if entity in tmp:
                entity_in_entity = True
                break
        if entity_in_entity:
            continue

        # if (
        #     entity.lower() == data1[qidx]["target_answer_type"].lower()
        #     and len(entity.split(' ')) <= 1
        # ):
        #     continue

        choice = False

        if info['label'] == 'gpt' and cnt_gpt < 5:
            choice = True
            cnt_gpt += 1

        else:
            choice = True
            cnt += 1
            
        if choice:
            # if len(get_longest_capitalized_substrings(question)) > 0:
            #     print(f"{entity} --- {info['label']} {info['bio_type']} {info['source'][0]}\n")

            printed_array.append(entity)
            if entity.lower() not in data_search_lower:
                data_search.append(entity.lower())
                data_search_lower.append(entity.lower())
                type.append(info['label'])
                
                merge_dict[qidx]["extracted_entities"]["detail"][entity] = {
                    'label': info['label'],
                    'bio_type': info['bio_type'],
                    'source': info['source']
                }

data_search_sorted = sorted(data_search, key=str.lower)
search_spacy = list(load_json(r'data\cache\mapping\entity2wiki.json').keys())

search_update = list(set(data_search_sorted)-set(search_spacy))
save_json(search_update, r'data\ner\data_search_9000_seq.json')

# save_json(data_search_sorted, r'data\ner\data_search_9000.json')


# save_json(result_dict, 'dict_filter_entity_extraction.json')

save_json(merge_dict, r'data\ner\entity_extraction_9000_seq.json')

# print(f"✅ Done. Saved to {OUTPUT_TXT}")
