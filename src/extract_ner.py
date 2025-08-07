from utils.utils import *

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
