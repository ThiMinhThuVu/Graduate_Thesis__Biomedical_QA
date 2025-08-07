import os
import spacy
from utils.utils import *
import pickle
from tqdm import tqdm
from src.decomposition import *
from src.retrieval import *
from src.extract_ner import *
from src.generate_answer import *
import pandas as pd

try:
    nlp_bio = spacy.load("en_ner_bc5cdr_md")  # Biomedical NER model
    print("Loaded biomedical NER model")
except:
    print("Biomedical NER model not found. Please install it with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz")
    nlp_bio = None

nlp_standard = spacy.load("en_core_web_lg")  # Standard English model

def generate_answer_for_subq(subquestion, search_kw, retrieval_information, step, paths, dataset="medhop"):
    if dataset in ["wiki", "cwq"]:
        final_context = f"Key Entities: {search_kw}\n\n" + f"{retrieval_information}"
        return generate_answer(subquestion, final_context, dataset=dataset)
    else:
        ner_filepath = paths['ner']
        output_ner = load_json(ner_filepath) if os.path.exists(ner_filepath) else {}
        step = str(step)
        
        if step in output_ner:
            print(f"Cache hit for NER step {step}.")
            bio_entities = output_ner[step]
        else:
            print(f"Extracting entities.")
            query_for_ner = subquestion + " " + retrieval_information
            bio_entities = extract_with_models(query_for_ner, nlp_bio, nlp_standard, dataset=dataset)
            output_ner[step] =bio_entities 
            save_json(output_ner, ner_filepath)

        final_context = f"Key Entities: {search_kw}\n\n" + f"{retrieval_information}"
        return generate_answer(subquestion, final_context, dataset=dataset)

def preprocess_sample(sample, settings, is_seq=False, dataset='medhop', type_q='compositional'):
    sample_id = str(sample['QIDX']) 
    
    if dataset == "medhop" and sample_id in settings['chrunk_mapping']:
        question = settings['chrunk_mapping'][sample_id]
    else:
        question = sample['Question']
        
    #DECOMPOSITION Question
    list_qi = question_decomposition(question, dataset=dataset, type_q=type_q)