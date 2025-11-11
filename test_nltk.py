from utils.process_wikidata_api import test

# import nltk

# nltk.download("stopwords")
# if nltk.__version__.startswith("3.9"): 
#     nltk.download("averaged_perceptron_tagger_eng") 
# else: 
#     nltk.download("averaged_perceptron_tagger")

# from src.gpt_request import *

# def gen_ans_multiple_choice(question, context, dataset='medhop', model = "o3-mini"):
#     prompt = f"""Question: {question}
#         Instructions:
#         - Please base your answer ONLY on the context provided.
#         - Please only choose the given answers (without additional explanations). For example: "A", "B", "C", "D", "E", "F", "G", ...
#         - Do not include anything else aside from your answer.

#         Context:
#         {context}
#         Please provide your short answer to the question based on the context above."""
#     if dataset == 'wiki':
#         messages = [
#             {"role": "system", "content": "You are an expert in general knowledge. Your task is to provide accurate answers based on the provided information. Please only output the choice (without additional explanations)"},
#             {"role": "user", "content": prompt}
#         ]
#     else:
#         messages = [
#             {"role": "system", "content": "You are an expert in medical and scientific knowledge. Your task is to provide accurate answers based on the provided information. Please only output the choice (without additional explanations)"},
#             {"role": "user", "content": prompt}
#         ]
    
#     result_json = request_api(messages, model)
    
#     return result_json

# question = "When is that person's date of birth?"
# q_relation = "date of birth"
# key_entity = "Emich, 5th Prince of Leiningen"
# context = "A. Emich, 5th Prince of Leiningen (description: Prince of Leiningen (1866-1939)) - date of birth: ['18 January 1866Gregorian'] (source: 1866-01-18T00:00:00Z)\n"
# res = gen_ans_multiple_choice(question + f", {key_entity}", context, dataset='wiki', model="o3-mini")
# print(f"Answer: {res}")