from utils.utils import *
from utils.abstract_wiki import crawl_abstract_wiki

CWQ_LABELS_EXPANDED = {
    'PERSON',      # Người, diễn viên, cầu thủ...
    'GPE',         # Địa điểm địa chính trị (Quốc gia, thành phố)
    'ORG',         # Tổ chức, công ty, trường học
    'LOC',         # Địa điểm không phải GPE (Sông, núi)
    'WORK_OF_ART', # Tên phim, sách, bài hát
    'EVENT',       # Sự kiện, cuộc thi, chiến tranh
    'FAC',         # Cơ sở vật chất (Tòa nhà, sân bay)
    'PRODUCT',     # Sản phẩm
    'MONEY',       # Tiền tệ
    'NORP',        # Quốc tịch, nhóm tôn giáo/chính trị (ví dụ: American, Catholic)
    'LANGUAGE'    
}

BIO_LABELS = {'DISEASE', 'CHEMICAL', 'DRUG', 'GENE', 'BODY_PART'}

def extract_with_models(query, nlp_bio, nlp_standard, dataset="medhop"):
    entities_set = set()

    target_labels = CWQ_LABELS_EXPANDED if dataset == "cwq" else BIO_LABELS

    doc_standard = nlp_standard(query)
    for ent in doc_standard.ents:
        if ent.label_ in target_labels:
            entities_set.add(ent.text)

    if dataset != "cwq" and nlp_bio:
        doc_bio = nlp_bio(query)
        for ent in doc_bio.ents:
            entities_set.add(ent.text)

    return list(entities_set)

def format_search_results(search_results, topk=5):
    context_parts = []
    for i, result in enumerate(search_results):            
        if i == topk:
            break
        
        context_parts.append(f"- {result['title']} {result['snippet']}")
                                       
    return "\n".join(context_parts)

def format_search_results(search_results, topk=5):
    context_parts = []
    for i, result in enumerate(search_results):            
        if i == topk:
            break
        
        context_parts.append(f"- {result['title']} {result['snippet']}")
                                       
    return "\n".join(context_parts)

def format_search_results_wiki_abs(search_results, step, filepath, topk=5):
    context_parts = []
    cnt = 0
    if os.path.exists(filepath):
        output_wiki = load_json(filepath)
        if step in output_wiki:
            for link, item in output_wiki[step]['dict_para'].items():
                if cnt == topk:
                    break
                if "List_"  in link:
                    continue                
                context_parts.append(f"- {item['title']}: {item['snippet']} {item['abstract']}")
                cnt += 1
                
            return "\n".join(context_parts)
    else:
        output_wiki = {}
        
    output_wiki[step] = {}
    output_wiki[step]["dict_para"] = {}
    
    for i, result in enumerate(search_results):            
        if i == topk:
            break
        title = result['title']
        url = result['link']
        if "List_" in url:
            continue
        abs_wiki = crawl_abstract_wiki(title, url)
        context_parts.append(f"- {title}: {result['snippet']} {abs_wiki}")
        
        output_wiki[step]["dict_para"][result['link']] = {'title': title,
                                                        'snippet': result['snippet'],
                                                        'abstract': abs_wiki}
    save_json(output_wiki, filepath)
    return "\n".join(context_parts)