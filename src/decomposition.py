from src.gpt_request import request_api
from src.prompt import load_prompt
from src.split_question import split_question
from utils.utils import load_json, save_json
import json
import os 

def gen_subquestion(question, is_seq=True, dataset='wiki', type_q='compositional'):
    messages = load_prompt(question, is_seq=is_seq, dataset=dataset, type_q=type_q)

def question_decomposition(question, dataset='wiki', type_q='compositional'):
    if dataset != 'wiki':
        return [question]
    
    if type_q in ['comparison', 'bridge_comparison']:
        return split_question(question)
    else:
        return [question]
        
    