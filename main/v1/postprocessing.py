import spacy
import re
import pandas as pd
import unicodedata

non_bio = spacy.load("en_core_web_lg") 

def normalize_verbal_numbers(text):
    for word, mul in MULTIPLIERS.items():
        m = re.search(r"(\d+(?:\.\d+)?)\s+" + word, text)
        if m:
            num = float(m.group(1))
            return str(int(num * mul))
    return text

def extract_acronym(text: str) -> str:
    words = text.split()
    letters = [w[0] for w in words if w[0].isalpha()]
    return "".join(letters)

MULTIPLIERS = {
    "million": 1_000_000,
    "billion": 1_000_000_000
}

def extract_parenthesis_candidates(text: str):
    """
    Return list of candidate strings from A (B)
    """
    candidates = []
    text = text.strip()
    
    m = re.search(r"^(.*?)\s*\((.*?)\)\s*$", text)
    if not m:
        return [text]
    
    main, inner = m.group(1), m.group(2)
    candidates.append(main.strip())
    
    # reject numeric-only or measurement
    # if not re.search(r"\d", inner):
    candidates.append(inner.strip())
    
    return candidates

def normalize_unicode(text: str) -> str:
    if not text:
        return text
    
    text = text.lower()
    # unicode normalize
    text = unicodedata.normalize("NFKD", text)
    
    # remove diacritics (ü → u, ä → a, …)
    text = "".join(c for c in text if not unicodedata.combining(c))
     
    return text

def normalize_estimation_question(text: str) -> str:
    patterns = [
        r"\b(over|under|around|about|approximately|approx|nearly|roughly|almost|more than|less than|at least|at most)\b",
        r"\b(years?|months?|weeks?|days?|decades?|centur(?:y|ies)|recent years|past years|last years)\b",
        r"%"
    ]
    
    text = text.lower()
    for p in patterns:
        text = re.sub(p, " ", text)
        
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def yes_no_type(answer):
    return answer.lower() in ["yes", "no"]

def chromosome_type(answer):
    return re.match(r"chromosome \d+", answer, re.IGNORECASE) is not None

def gene_type(answer):
    if 'gene' in answer.lower() or 'gene mutation' in answer.lower() or 'mutation' in answer.lower():
        return True
    else:
        return False

def syndrome_type(answer):
    if 'syndrome' in answer.lower():
        return True
    else:
        return False

def how_many_type(answer):
    pattern = r'^\s*\d+(?:\.\d+)?\s+(?:[a-zA-Zμ%°/]+(?:\s+[a-zA-Zμ%°/]+)*)\s*$'
    return bool(re.match(pattern, answer.strip()))

def number_type(answer):
    return answer.isdigit()

def identify_entity_type(answer):
    if yes_no_type(answer):
        return 'yes_no'
    elif chromosome_type(answer):
        return "chromosome"
    elif gene_type(answer):
        return "gene"
    elif how_many_type(answer):
        return "how many"
    elif number_type(answer):
        return "number"
    elif syndrome_type(answer):
        return 'syndrome'
    else:
        answer = answer.strip()
        doc_non_bio = non_bio(answer)  # Đưa toàn bộ answer vào model non_bio
        for ent in doc_non_bio.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"]:
                return ent.label_
            
def extract_quantity(text):
    pattern = r'^\s*(\d+(?:\.\d+)?)\s+(?:[a-zA-Zμ%°/]+(?:\s+[a-zA-Zμ%°/]+)*)\s*$'
    match = re.match(pattern, text.strip())
    return match.group(1) if match else None


def remove_titles(answer):
    titles = r'\b(Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Madam|Mx)\.?\s*'
    return re.sub(titles, '', answer, flags=re.IGNORECASE).strip()


def is_abbreviation(full_form: str, abbr: str) -> bool:
    tokens = re.findall(r"\w+", full_form, flags=re.UNICODE)
    initials = "".join(token[0] for token in tokens)
    return initials.upper() == abbr.upper()

def extract_preferred(text: str) -> str:
        match = re.match(r"^(.*?)\s*\((\w+)\)\s*$", text)
        if not match:
            return text.strip()

        full_form = match.group(1).strip()
        abbr = match.group(2).strip()

        if is_abbreviation(full_form, abbr):
            return abbr
        else:
            return full_form
        
        
def postprocessing(answer):
    ans_type = identify_entity_type(answer)

    format_answer = answer.strip('\'".,')

    if ans_type == 'gene':
        format_answer = format_answer.replace(' gene mutation', '').replace(' gene', '').replace(' mutation', '')

    elif ans_type == 'chromosome':
        match = re.match(r"(Chromosome \d+)", format_answer, re.IGNORECASE)
        format_answer = match.group(1)

    elif ans_type == 'how_many':
        format_answer = extract_quantity(format_answer)

    elif ans_type == 'PERSON':
        format_answer = remove_titles(format_answer)

    format_answer = extract_preferred(format_answer)
    return format_answer

# print(extract_parenthesis_candidates("mammalian and yeast GSs (GT3 family)"))