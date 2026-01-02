from typing import Optional, List, Dict, Any
import json

BIO_LABELS_LIST = ["DISEASE", "CHEMICAL", "DRUG", "GENE", "BODY_PART", "ORG", "GPE"]
WIKI_LABELS_LIST = ["PERSON", "WORK_OF_ART", "GPE", "ORG", "LOC", "EVENT", "CARDINAL", "NORP", "LANGUAGE"]
    
def load_prompt_gen_sub(question=None, is_seq=False, dataset=None, type_q=None, context=None):
    if question:
        if is_seq:
            messages = _load_subquestion_prompt(question, dataset, type_q)
        else:
            messages = _load_search_keyword_prompt(question, dataset, type_q)            
      
    return messages

def load_prompt_gen_ans(question, key_entities, relevant_info, dataset=None, type_q=None):
    if dataset == "medhop":
        prompt = prompt_gen_answer_medhop_step_3b(question, key_entities, relevant_info)
    elif dataset == "wiki":
        prompt = prompt_gen_answer_wiki_step_3b(question, key_entities, relevant_info, type_q=type_q)

    return prompt

def _load_search_keyword_prompt(question, dataset=None, type_q=None):
    if dataset == 'medhop':
        labels_str = ", ".join(BIO_LABELS_LIST)
        messages = [
            {
        "role": "system",
        "content": f"""You are an expert in generating ONLY 1 concise and effective Google search query to help answer a complex biomedical question.
    Your goal is to:
    1. Provide a single, keyword-based search query string that best targets the core information needed from the original question.
    2. Identify a list of target entity types (target_labels) that need to be extracted from the context to answer the question. The "target_labels" MUST be selected ONLY from this list: [{labels_str}].
    3. Identify a list of specific, concrete, and retrievable entities explicitly mentioned or clearly implied in the question. 

    Key principles for the single query:
    1. The query should be a distillation of the original complex question, focusing on essential keywords.
    2. The query MUST be concise, using specific keywords INCLUDING IMPORTANT BIOMEDICAL KEYWORD, nouns, and technical terms critical to answering the question.
    3. The "target_labels" should contain the entity type(s) that represent the expected answer or the core subject of the question.
    4. The "key_entity" list of specific. Exclude generic or abstract concepts that are not useful for information retrieval (e.g., “medication”, “drug”, “treatment”).
         
    The final output MUST be a JSON array containing a single object. This object must contain a single key "search_keyword", whose value is the Google search query string and "target_labels" (a list of strings) and "key_entity".
    """
            },
            {
        "role": "user",
        "content": f"""Here are 5 examples demonstrating the desired JSON output format, which is a JSON array containing a single object with a "search_keyword" field holding the Google search query string.

    ---
    **Example 1:**

    Question: "Which gene containing a helicase domain is mutated in some patients with Smith-Fineman-Myers syndrome?"

    JSON desired output format:
    [
        {{ 
            "search_keyword": "Smith-Fineman-Myers syndrome mutated gene helicase domain",
            "target_labels": ["GENE"],
            "key_entity": [
                "Smith-Fineman-Myers syndrome",
                "helicase domain"
            ]
        }}
    ]
    ---
    **Example 2:**

    Question: "Which lactate dehydrogenase tetramer is most common in the primary organ of the nervous system?"

    JSON desired output format:
    [
        {{ 
            "search_keyword": "lactate dehydrogenase tetramer most common primary organ nervous system",
            "target_labels": ["CHEMICAL"],
            "key_entity": [
                "lactate dehydrogenase tetramer",
                "nervous system"
            ]
        }}
    ]
    ---
    **Example 3:**

    Question: "Which physician, along with Edvard Ehlers, is the Ehlers Danlos syndromes named after?"

    JSON desired output format:
    [
        {{ 
            "search_keyword": "Ehlers Danlos syndromes named after along with Edvard Ehlers",
            "target_labels": ["DISEASE"],
            "key_entity": [
                "Ehlers-Danlos syndrome",
                "Edvard Ehlers"
            ]            
        }}
    ]
    ---
    **Example 4:**

    Question: "Which benign, autosomal recessive hyperbilirubinemia syndrome is caused by a mutation in the UGT1A1 gene on chromosome 2?"

    JSON desired output format:
    [
        {{ 
            "search_keyword": "benign autosomal recessive hyperbilirubinemia UGT1A1 gene mutation chromosome 2",
            "target_labels": ["DISEASE", "GENE"],
            "key_entity": [
                "benign autosomal recessive hyperbilirubinemia",
                "UGT1A1 gene",
                "chromosome 2"
            ]
        }}
    ]
    ---
    **Example 5:**

    Question: "Which recognized type of osteochondritis is characterized by uneven vertebral growth and kyphosis that typically develops during adolescence?"

    JSON desired output format:
    [
        {{ 
            "search_keyword": "osteochondritis type uneven vertebral growth kyphosis adolescence",
            "target_labels": ["DISEASE"],
            "key_entity": [
                "osteochondritis",
                "vertebral growth",
                "kyphosis",
                "adolescence"
            ]
        }}
    ]
    ---

    Now, process the following:

    Question: "{question}"

    Task: Generate ONLY 1 effective Google search query that distills the core of the given question.

    Return a JSON array with "search_keyword" and "target_labels" (choosing from [{labels_str}]). For example:
    [
        {{ 
            "search_keyword": "...",
            "target_labels": [...],
            "key_entity": [...],
        }}
    ]
    """
            }
        ]
    elif dataset == 'wiki':
        if type_q == "comparison":
            labels_str = ", ".join(WIKI_LABELS_LIST)
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert in generating ONLY 1 concise and effective Google search query to help answer a complex question.
                    Your goal is to provide a single, keyword-based query string that best targets the core information needed from the original question.

                    Key principles for the single query:
                    1. The query should be a distillation of the original complex question, focusing on essential keywords, proper names, and the core relationship being asked about.
                    2. The query MUST be concise, using specific nouns and terms critical to answering the question, and should omit conversational parts like "Who is the".
                    3. The key entity is the main subject (person, place, thing, or concept) the question is about.
                    4. "target_labels": A list of entity types to extract to answer the question, selected ONLY from: [{labels_str}].
                    5. **Adapt relation keywords based on the key entity's type.** This is crucial for avoiding ambiguity. For example:
                    - For a **Person** (Human), use specific relations like: `country of citizenship`, `date of birth`, `occupation`,...
                    - For a **Creative Work** (Film, Album, Song), use relations like: `country of origin`, `publication date`, `director`, `producer`,...
                    - For a **Location or Organization** (Building, Museum, University, Mountain), use relations like: `country`, `inception`,...

                    The final output MUST be a JSON array containing a single object. This object must contain two keys:
                    - "search_keyword": The generated Google search query string.
                    - "key_entity": The primary subject extracted from the question.
                    - "target_labels": choosing from [{labels_str}]
                    """
                },
                {
                    "role": "user",
                    "content": f"""Here are 5 examples demonstrating the desired JSON output format, which is a JSON array containing a single object with a "search_keyword" and "key_entity" field.

                    ---
                    **Example 1:**

                    Question: "Who directed the (1957 Film) 'Just Once A Great Lady'?"

                    JSON desired output format:
                    [
                        {{
                            "search_keyword": "director 'Just Once A Great Lady (1957 Film)'",
                            "key_entity": "Just Once A Great Lady (1957 Film)",
                            "target_labels": ["PERSON"]
                        }}
                    ]
                    ---
                    **Example 2:**

                    Question: "In which country is the film Carnival Of Souls from?"

                    JSON desired output format:
                    [
                        {{ 
                            "search_keyword": "'Carnival Of Souls' country of origin",
                            "key_entity": "Carnival Of Souls",
                            "target_labels": ["GPE"]
                        }}
                    ]
                    ---
                    **Example 3:**

                    Question: "When was the film Engal Aasan released?"

                    JSON desired output format:
                    [
                        {{
                            "search_keyword": "'Engal Aasan' publication date",
                            "key_entity": "Engal Aasan",
                            "target_labels": ["DATE"]
                        }}
                    ]
                    ---
                    **Example 4:**

                    Question: "When was Michelle Tong (Writer) born?"

                    JSON desired output format:
                    [
                        {{
                            "search_keyword": "'Michelle Tong (Writer)' date of birth",
                            "key_entity": "Michelle Tong (Writer)",
                            "target_labels": ["DATE"]
                        }}
                    ]
                    ---
                    **Example 5:**

                    Question: "What is the nationality of the writer William C. White?"

                    JSON desired output format:
                    [
                        {{
                            "search_keyword": "'William C. White' country of citizenship",
                            "key_entity": "William C. White",
                            "target_labels": ["NORP"]
                        }}
                    ]
                    ---
                    Now, process the following:
                    Question: "{question}"
                    Task: Generate ONLY 1 effective Google search query that distills the core of the given question.
                    Return a JSON array with "search_keyword", "key_entity", and "target_labels" (choosing from [{labels_str}]), following the format of the examples above.
                    [
                        {{ 
                            "search_keyword": "...",
                            "key_entity": "...",
                            "target_labels": "..."
                        }}
                    ]
                    """
                }
            ]
        else:
            messages = None

    return messages

def _load_subquestion_prompt(question, dataset=None, type_q=None):
    if dataset == 'medhop':
        labels_str = ", ".join(BIO_LABELS_LIST)
        messages = [
            {
        "role": "system",
        "content": f"""You are a highly specialized AI assistant, an expert in biomedical knowledge, skilled at decomposing complex biomedical questions into a sequence of simpler, interconnected natural language sub-questions. Your primary goal is to create a logical, step-by-step plan that faithfully represents the reasoning process to answer the main question.

        **Core Instructions for Decomposition:**

        0.  **Faithful Reasoning via Decomposition**: Your primary task is to externalize the reasoning process by breaking the `Original Complex Question` into a series of sub-questions. This decomposition itself forms the transparent and explainable path to the solution.
        1.  **Step-by-Step Plan**: Generate a sequence of sub-questions.
        2.  **Self-Contained and Searchable Sub-questions**: Each sub-question MUST be formulated so it is **self-contained and understandable in isolation**. This means it must include all necessary context from the original question or previous steps to be answerable by a standard search engine query *without needing to refer back to the original complex question or other sub-questions implicitly*. Sub-questions must be specific and unambiguous. For example, avoid pronouns or references that don't have explicit antecedents within the sub-question itself.
            * **Prefer Single Entities:** When a sub-question's answer is used in a later step, it should ideally yield a **single, specific entity** (e.g., one specific gene name, one particular symptom, a single defined mechanism). This ensures that natural language references to this entity in subsequent steps (e.g., 'that gene', 'the identified symptom') are unambiguous.
            * **Explicit List Handling:** If an intermediate step *must* produce a list of items because the nature of the query requires it (e.g., "What known [genes] are associated with X?"), any subsequent sub-question using that list **must explicitly state how to process it** (e.g., "Which of the [previously identified genes] are also [specific characteristic, e.g., 'expressed in the liver']?" or "What is the most frequently implicated [gene] from [the list found in the previous step] for [specific context]?").    * **Avoid Vague List Generation for Intermediate Steps:** Do not formulate intermediate sub-questions that ask for "some examples" or an undefined set of items if that result is intended to be fed precisely into a subsequent step, unless that subsequent step clearly outlines how to operate on such a potentially open-ended or non-exhaustive list.
        3.  **Logical Dependency**:
            * The answer to one sub-question will often be used to formulate the subsequent sub-question.
            * When a sub-question depends on the result of a previous step, this relationship should be conveyed by using explicit natural language references to the entity or information from the previous step within the current sub-question. The sub-questions should form a logical chain where context flows naturally from one step to the next. Each sub-question should still aim to be as self-contained as possible by incorporating necessary context.
        4.  **Sub-question Properties**: For each step in your decomposition:
            * **`subquestion`**: State the sub-question in clear, natural language, adhering to the self-contained principle and clarity of result.
            * **`search_keyword`**: Provide a concise, keyword-based search query that captures the essence of the sub-question. This should be a distilled version of the sub-question, focusing on the key biomedical terms and concepts necessary to retrieve relevant information. The `search_keyword` should be a direct, searchable query that can be used to find the answer to the sub-question. It should not include unnecessary words or context that does not contribute to the search.
            * **`target_labels`**: A list of entity types to extract to answer the question, selected ONLY from: [{labels_str}].
            * **`key_entity`**: Identify a list of specific, concrete, and retrievable entities explicitly mentioned or clearly implied in the question. Exclude generic or abstract concepts that are not useful for information retrieval (e.g., “medication”, “drug”, “treatment”).
        5.  **Efficiency and Necessity & Defining Question Types**:
            * Decompose the question into the **minimum number of steps absolutely necessary** to logically arrive at the final answer.
            * **Multi-step Question Definition**: A multi-step question typically involves a chain of information retrieval where intermediate entities or properties must be resolved sequentially. One piece of information is needed to find the next, eventually leading to the final answer.
                * *Example of a multi-step question requiring decomposition*: "Which human chromosome contains the gene that encodes the enzyme that can inactivate antithrombin?" (This requires finding: 1. The enzyme, 2. The gene encoding it, 3. The chromosome containing that gene).
            * **Single-step Question Definition**: A single-step question usually asks for a primary entity or concept (e.g., a "term", "name", "definition", "cause", "symptom") that can be identified directly from a set of given characteristics, even if those characteristics are numerous or include other named entities. If all characteristics defining the target entity can be reasonably combined into a single, direct, and searchable query that is likely to yield the answer, a single step is preferred. Avoid intermediate steps that merely define well-known qualifying characteristics/entities unless defining them is absolutely essential to resolve ambiguity in the main entity being sought or if the characteristic itself is the primary unknown blocking the answer. The goal is to find the most direct path to the answer.
            * If the original complex question fits the single-step definition and can be directly and unambiguously answered by a single, well-formulated search query, provide a single-step decomposition.
            * Avoid creating sub-questions that are overly trivial, redundant, or that primarily restate parts of the original question that could be resolved within a more encompassing single or earlier step. Each step must isolate a *distinct and necessary piece of unknown information* crucial for the overall answer.
        6.  **Final Answer Derivation**: The final sub-question's purpose is to synthesize the information gathered from all preceding, answered sub-questions to directly and comprehensively answer the `Original Complex Question`. The decomposition should lead to a state where this final step is a clear derivation from the sub-answers.
        7.  **Output Format**: **YOU MUST USE JSON FORMAT ONLY.** Ensure the entire output is a valid JSON array of step objects.
     
        Original Complex Question: '{question}'
        """
            },
            {
            "role": "user",
            "content": f"""
        Here are 5 examples demonstrating the desired JSON output format and decomposition logic:

        ---
        **Example 1:**
        "Which human chromosome contains the gene that encodes the enzyme that can inactivate antithrombin?"

        JSON desired output format:
        [
            {{
                "subquestion": "Which gene encode enzyme can inactivate antithrombin?",
                "search_keyword": "gene encode enzyme inactivate antithrombin",
                "target_labels": ["GENE"],
                "key_entity": ["antithrombin"]
            }},
            {{
                "subquestion": "Which human chromosome contains that gene?",
                "search_keyword": "human chromosome gene encode enzyme inactivate antithrombin",
                "target_labels": ["BODY_PART"],
                "key_entity": []
            }}
        ]

        ---
        **Example 2:**
        "Which lactate dehydrogenase tetramer is most common in the primary organ of the nervous system?"

        JSON desired output format:
        [
            {{
                "subquestion": "What is the primary organ of the human nervous system?",
                "search_keyword": "primary organ human nervous system",
                "target_labels": ["BODY_PART"],
                "key_entity": ["nervous system"]
            }},
            {{
                "subquestion": "Which specific lactate dehydrogenase tetramer is most commonly found in that organ?",
                "search_keyword": "lactate dehydrogenase tetramer most common primary organ nervous system",
                "target_labels": ["CHEMICAL"],
                "key_entity": []
            }}
        ]

        ---
        **Example 3:**
        "Which gene mutation is responsible for the syndrome characterized by flattened and deformed vertebrae, developmental delay, and lens coloboma?"

        JSON desired output format:
        [
            {{
                "subquestion": "Which syndrome is characterized by flattened and deformed vertebrae, developmental delay, and lens coloboma?",
                "search_keyword": "syndrome flattened deformed vertebrae developmental delay lens coloboma",
                "target_labels": ["DISEASE"],
                "key_entity": [
                    "flattened vertebrae",
                    "deformed vertebrae",
                    "developmental delay",
                    "lens coloboma"
                ]
            }},
            {{
                "subquestion": "Which gene mutation is responsible for that syndrome?",
                "search_keyword": "gene mutation syndrome flattened deformed vertebrae developmental delay lens coloboma",
                "target_labels": ["GENE"],
                "key_entity": []
            }}
        ]

        ---
        **Example 4:**
        "Which London suburb was home to the pioneering institutions that influenced John Langdon Down's work at Earlswood Asylum?"

        JSON desired output format:
        [
            {{
                "subquestion": "Which pioneering institutions influenced John Langdon Down's work at Earlswood Asylum?",
                "search_keyword": "pioneering institutions John Langdon Down Earlswood Asylum",
                "target_labels": ["ORG"],
                "key_entity": [
                    "John Langdon Down",
                    "Earlswood Asylum"
                ]
            }},
            {{
                "subquestion": "In which London suburb was that pioneering institution located?",
                "search_keyword": "London location pioneering institution John Langdon Down Earlswood Asylum",
                "target_labels": ["GPE"],
                "key_entity": []
            }}
        ]

        ---
        **Example 5:**
        "On which chromosome is the gene responsible for Carpenter syndrome, which is associated with brachydactyly, located?"

        JSON desired output format:
        [
            {{
                "subquestion": "Which gene is responsible for Carpenter syndrome?",
                "search_keyword": "gene responsible Carpenter syndrome",
                "target_labels": ["GENE"],
                "key_entity": [
                    "Carpenter syndrome"
                ]
            }},
            {{
                "subquestion": "On which chromosome is that gene located?",
                "search_keyword": "chromosome location gene",
                "target_labels": ["BODY_PART"],
                "key_entity": []
            }}
        ]


        Now, please provide the decomposition with the minimum number of necessary steps for my original complex question.
        Remember the strict formatting requirements:
        1.  Entire output MUST be a valid JSON array.
        2.  Ensure all sub-questions are self-contained and directly contribute to answering the original complex question, adhering to the guidelines for intermediate results.

        Here is the original complex question: "{question}"
        """
                }
            ]
    elif dataset == "wiki":
        labels_str = ", ".join(WIKI_LABELS_LIST)
        if type_q == "bridge_comparison":
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a highly specialized AI assistant, an expert at decomposing complex questions into a sequence of simpler, interconnected natural language sub-questions. Your primary goal is to create a logical, step-by-step plan that faithfully represents the reasoning process to answer the main question.

            **Core Instructions for Decomposition:**

            0.  **Faithful Reasoning via Decomposition**: Your primary task is to externalize the reasoning process by breaking the `Original Complex Question` into a series of sub-questions. This decomposition itself forms the transparent and explainable path to the solution.
            1.  **Step-by-Step Plan**: Generate a sequence of sub-questions.
            2.  **Dynamic and Dependent Sub-questions**: Your sub-questions form a dynamic plan that will be executed step-by-step, where the results of one step are used in the next.
                *Sub-question Formulation*: A sub-question can be either fully self-contained or a template that depends on the answer from a previous step.  
                When it's dependent, use a clear, general placeholder (e.g., "that director," "that person," "the entity found in the previous step") to indicate where the answer from the previous step will be programmatically inserted.
                *Search Keyword Formulation*: Similarly, the search_keyword can be a complete, standalone query or a template (e.g., "director date of birth," "person spouse"). This template is designed to be combined with the result of a prior step to create the final, precise search query. This structure creates a powerful, multi-step reasoning plan.
                *target_labels*: A list of entity types to extract to answer the question, selected ONLY from: [{labels_str}].
            3.  **Logical Dependency**:
                * The answer to one sub-question will often be used to formulate the subsequent sub-question.
                * When a sub-question depends on the result of a previous step, this relationship should be conveyed by using explicit natural language references to the entity or information from the previous step within the current sub-question.
            4.  **Sub-question Properties**: For each step in your decomposition:
                * **`subquestion`**: State the sub-question in clear, natural language.
                * **`search_keyword`**: Provide a concise, keyword-based search query that captures the essence of the sub-question.
            5.  **Efficiency and Necessity**:
                * Decompose the question into the **minimum number of steps absolutely necessary** to logically arrive at the final answer.
                * If the original complex question can be directly and unambiguously answered by a single, well-formulated search query, provide a single-step decomposition.
            6.  **Output Format**: **YOU MUST USE JSON FORMAT ONLY.** Ensure the entire output is a valid JSON array of step objects.

            Original Complex Question: '{question}'
            """
                },
                {
                    "role": "user",
                    "content": f"""
            Here are 6 examples demonstrating the desired JSON output format and decomposition logic.

            ---
            **Example 1:**
            "What is the country of the director of film Move (1970 Film)"

            JSON desired output format:
            [
                {{
                    "subquestion": "What is the director of film Move (1970 Film)?",
                    "search_keyword": "director 'Move (1970 Film)'",
                    "key_entity": "Move (1970 Film)",
                    "target_labels": ["PERSON"]
                }},
                {{
                    "subquestion": "What country of citizenship is that film director from?",
                    "search_keyword": "country of citizenship",
                    "key_entity": "",
                    "target_labels": ["GPE"]
                }}
            ]
            ---
            **Example 2:**
            "When was the director of film Blindfold (1966 Film) born?"

            JSON desired output format:
            [
                {{
                    "subquestion": "Who directed the (1966 Film) 'Blindfold'?",
                    "search_keyword": "director 'Blindfold (1966 Film)'",
                    "key_entity": "Blindfold (1966 Film)",
                    "target_labels": ["PERSON"]
                }},
                {{
                    "subquestion": "When was that person born?",
                    "search_keyword": "date of birth",
                    "key_entity": "",
                    "target_labels": ["DATE"]
                }}
            ]
            ---
            **Example 3:**
            "What is the date of death of Helen F. Holt's husband?"

            JSON desired output format:
            [
                {{
                    "subquestion": "Who was the husband of Helen F. Holt?",
                    "search_keyword": "'Helen F. Holt' husband",
                    ""key_entity": "Helen F. Holt",
                    "target_labels": ["PERSON"]                    
                }},
                {{
                    "subquestion": "What is the date of death of that person?",
                    "search_keyword": "date of death",
                    "key_entity": "",
                    "target_labels": ["DATE"]
                }}
            ]
            ---
            **Example 4:**
            "Where did the director of film The Right Of The Strongest die?"

            JSON desired output format:
            [
                {{
                    "subquestion": "Who was the director of the film 'The Right Of The Strongest'?",
                    "search_keyword": "director film 'The Right Of The Strongest'",
                    "key_entity": "The Right Of The Strongest",
                    "target_labels": ["PERSON"]
                    
                }},
                {{
                    "subquestion": "Where did that person die?",
                    "search_keyword": "place of death",
                    "key_entity": "",
                    "target_labels": ["GPE"]
                }}
            ]
            ---
            **Example 5:**
            "Who is the spouse of the director of film Just Once A Great Lady (1957 Film)?"

            JSON desired output format:
            [
                {{
                    "subquestion": "Who directed the 'Just Once A Great Lady (1957 Film)'?",
                    "search_keyword": "director 'Just Once A Great Lady (1957 Film)'",
                    "key_entity": "Just Once A Great Lady (1957 Film)",
                    "target_labels": ["PERSON"]
                }},
                {{
                    "subquestion": "Who was the spouse of that person?",
                    "search_keyword": "spouse of",
                    "key_entity": "",
                    "target_labels": ["PERSON"]
                }}
            ]
            ---
            **Example 6:**
            "When was the producer of "My Friends Told Me About You" born?"
            JSON desired output format:
            [
                {{
                    "subquestion": "Who produced the 'My Friends Told Me About You'?",
                    "search_keyword": "producer 'My Friends Told Me About You'",
                    "key_entity": "My Friends Told Me About You",
                    "target_labels": ["PERSON"]
                }},
                {{
                    "subquestion": "When was that person born?",
                    "search_keyword": "date of birth",
                    "key_entity": "",
                    "target_labels": ["DATE"]
                }}
            ]
            ---
            Now, please provide the decomposition with the minimum number of necessary steps for my original complex question.
            Remember the strict formatting requirements:
            1.  The entire output MUST be a valid JSON array.
            2.  Ensure all sub-questions are self-contained and directly contribute to answering the original complex question.

            Here is the original complex question: "{question}"
            """
                }
            ]
        elif type_q == "compositional":
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a highly specialized AI assistant. Your expertise is in analyzing questions that contain hidden sequential logic, particularly those involving relationships within social or organizational structures (e.g., family trees, corporate hierarchies). Your primary goal is to decompose these complex questions into a sequence of simpler, interconnected natural language sub-questions, creating a logical, step-by-step plan that reveals the reasoning process required to find the answer.
                
                    **Core Instructions for Decomposition:**
                    0.  **Faithful Reasoning via Decomposition**: Your primary task is to externalize the reasoning process by breaking the `Original Complex Question` into a series of sub-questions. This decomposition itself forms the transparent and explainable path to the solution.
                    1.  **Step-by-Step Plan**: Generate a sequence of sub-questions.
                    2.  **Dynamic and Dependent Sub-questions**: Your sub-questions form a dynamic plan that will be executed step-by-step, where the results of one step are used in the next.
                    *Sub-question Formulation*: A sub-question must be a template that depends on the answer from a previous step.
                    *When it's dependent, use a clear, general placeholder (e.g., "that director," "that person," "the entity found in the previous step", ...) to indicate where the answer from the previous step will be programmatically inserted.
                    * When a sub-question depends on the result of a previous step, this relationship should be conveyed by using explicit natural language references to the entity or information from the previous step within the current sub-question.
                    *Search Keyword Formulation*: Similarly, the search_keyword is a template (e.g., "director date of birth," "person spouse"). This template is designed to be combined with the result of a prior step to create the final, precise search query. This structure creates a powerful, multi-step reasoning plan.
                    *target_labels*: A list of entity types to extract to answer the question, selected ONLY from: [{labels_str}].
                    3.  **Logical Dependency**:
                    * The answer to one sub-question will be used to formulate the subsequent sub-question.
                    * When a sub-question depends on the result of a previous step, this relationship should be conveyed by using explicit natural language references to the entity or information from the previous step within the current sub-question.
                    4.  **Sub-question Properties**: For each step in your decomposition:
                    * **`subquestion`**: State the sub-question in clear, natural language.
                    * **`search_keyword`**: Provide a concise, keyword-based search query that captures the essence of the sub-question.
                    5.  **Efficiency and Necessity**:
                    * Decompose the question into the **minimum number of steps absolutely necessary** to logically arrive at the final answer.
                    * If the original complex question can be directly and unambiguously answered by a single, well-formulated search query, provide a single-step decomposition.
                    6.  **Output Format**: **YOU MUST USE JSON FORMAT ONLY.** Ensure the entire output is a valid array of step objects.

                    Original Complex Question: '{question}'
                    """
                    },
                    {
                        "role": "user",
                        "content": f"""
                    Here are 4 examples demonstrating the desired JSON output format and decomposition logic.
                    ---
                    **Example 1:**
                    "Which award did the director of the movie Parasite receive at the Oscars?"
                    JSON desired output format:
                    [
                    {{
                        "subquestion": "Who is the director of the movie Parasite?",
                        "search_keyword": “director of the movie 'Parasite'",
                        "key_entity": “Parasite",
                        "target_labels": ["PERSON"]
                    }},
                    {{
                        "subquestion": "Which award did that director receive at the Oscars?",
                        "search_keyword": "award receive at Oscars",
                        "key_entity": "",
                        "target_labels": ["WORK_OF_ART"]
                    }}
                    ]

                    ---
                    **Example 2:**
                    When is Ada Lovelace (1815–1852)'s father birthday?
                    JSON desired output format:
                    [
                    {{
                        "subquestion": "Who was the father of Ada Lovelace (1815–1852)?",
                        "search_keyword": "Ada Lovelace (1815–1852) father",
                        "key_entity": "Ada Lovelace (1815–1852)",
                        "target_labels": ["PERSON"]
                    }},
                    {{
                        "subquestion": "When is that person date of birth?",
                        "search_keyword": "date of birth",
                        "key_entity": "",
                        "target_labels": ["DATE"]
                    }}
                    ]
                    ---
                    **Example 3:**
                    "Where was the director of film Doctor Strange (2016 film) born"
                    JSON desired output format:
                    [
                    {{
                        "subquestion":"Who was the director of film Doctor Strange (2016 film)",
                        "search_keyword": "director of film 'Doctor Strange (2016 film)'",
                        "key_entity": "Doctor Strange (2016 film)",
                        "target_labels": ["PERSON"]
                    }},
                    {{
                        "subquestion": "Where was director born?",
                        "search_keyword": "place of birth",
                        "key_entity": "",
                        "target_labels": ["GPE"]
                    }}
                    ]
                    ---
                    **Example 4:**
                    "Which country the director of film Web Of Suspicion is from?"
                    JSON desired output format:
                    [
                    {{
                        "subquestion":"Who was the director of film Web Of Suspicion?",
                        "search_keyword": "director of film 'Web Of Suspicion'",
                        "key_entity": "Web Of Suspicion",
                        "target_labels": ["PERSON"]
                    }},
                    {{
                        "subquestion": "Which country of citizenship is that director from?",
                        "search_keyword": "country of citizenship",
                        "key_entity": "",
                        "target_labels": ["GPE"]
                    }}
                    ]
                    ---
                    Now, please provide the decomposition with the minimum number of necessary steps for my original complex question.
                    Remember the strict formatting requirements:
                    1.  The entire output MUST be a valid JSON array.
                    2.  Ensure each sub-question logically follows the previous one and directly contributes to answering the original complex question.

                    Here is the original complex question: "{question}"
                    """
                }
            ]
        elif type_q == "inference":
            messages = [
            {
                "role": "system",
                "content": """You are an expert in indentifying a key entity and key relation in a question.
            
            Key principles for identifying key entities and relations include:
            1. Look for the main subject and object in the question.
            2. Identify the action or relationship being asked.
            3. Consider the context and any additional information provided.
            4. target_labels: A list of entity types to extract to answer the question, selected ONLY from: [{labels_str}].

            The final output MUST be a JSON object. This object must contain keys: "key_entity", "key_relation", and "target_labels".
            """
            },
            {
                "role": "user",
                "content": f"""Here are the examples demonstrating the desired JSON output format, which is an array containing a single object with "key_entity" and "key_relation" fields:

            ---
            **Example 1:**

            Question: "Who is Giselda Volodi's sibling-in-law?"

            JSON desired output format:
            [
                {{
                    "search_keyword": "'Giselda Volodi' sibling-in-law",
                    "key_entity": "Giselda Volodi",
                    "key_relation": "sibling-in-law",
                    "target_labels": ["PERSON"]
                }}
            ]

            ---
            **Example 2:**

            Question: "Who is Humphrey Of Montfort's maternal grandmother?"

            JSON desired output format:
            [
                {{
                    "search_keyword": "'Humphrey Of Montfort' maternal grandmother",
                    "key_entity": "Humphrey Of Montfort",
                    "key_relation": "maternal grandmother",
                    "target_labels": ["PERSON"]
                }}
            ]

            ---
            **Example 3:**

            Question: "Who is the aunt of Margaretta Taylor?"

            JSON desired output format:
            [
                {{
                    "search_keyword": "'Margaretta Taylor' aunt",
                    "key_entity": "Margaretta Taylor",
                    "key_relation": "aunt",
                    "target_labels": ["PERSON"]
                }}
            ]

            ---
            **Example 4:**

            Question: "What is the field of work of Fred Diamond?"

            JSON desired output format:
            [
                {{
                    "search_keyword": "'Fred Diamond' field of work",
                    "key_entity": "Fred Diamond",
                    "key_relation": "field of work",
                    "target_labels": ["ORG"]
                }}
            ]

            ---

            Now, process the following:

            Question: "{question}"
            
            Return a JSON array containing a single object. This object must contain a single key "search_keyword", whose value is the Google search query string. For example:

            [
                {{
                    "search_keyword": "...",
                    "key_entity": "...",
                    "key_relation": "...",
                    "target_labels": "..."
                }}
            ]
            """
            }
        ]

    return messages


def prompt_gen_answer_medhop_step_3b(question, key_entities, context):
    prompt = f"""Question: {question}
        Instructions:
        1. Determine the question type from the question text.
        2. Generate two parts in JSON format:
        - long_answer: a detailed explanation based on the information.
        - short_answer: strictly follow these rules based on the question type:
        * For questions asking "how many" or about counts, return only a number or numeric phrase (e.g., "10", "ten").
        * For questions asking about time or duration, return a concise time value (e.g., "36 hours", "3 days").
        * For yes/no questions, return "Yes" or "No".
        * For questions starting with "Who", "Which", or "What" that ask for a specific entity (e.g., person, disease, drug, gene, chromosome), return the full noun phrase naming that entity.
            - Examples: "chromosome X", "interleukin-6", "dopamine receptor D2".
        * For questions asking about organism, return organism type (e.g., "bacterium", "fungus").

        3. Do not add any extra commentary or text outside the JSON.

        Information:  
        Key entities: {key_entities}
        Context: {context}

        Response format (strict JSON):
        {{  
        "long_answer": "[detailed explanation]",  
        "short_answer": "[short answer as per rules]"  
        }}
        """
    return prompt


def prompt_gen_answer_wiki_step_3b(question, key_entities, context, type_q=None):
    if type_q in ["bridge_comparison", "comparison"]:
        prompt = f"""Question: {question}
            Instructions:
            1. Analyze the question and the provided information to determine the answer.
            2. Generate a response in a single, valid JSON object with two keys: "long_answer" and "short_answer".
            - long_answer: A detailed explanation synthesized from the information.
            - short_answer: A single string with the direct answer. Strictly follow these rules:
            * **1. Yes/No Questions:** For questions requiring a "yes" or "no" answer, the answer must be "Yes" or "No".
            * **2. Default - Single Answer:** For all other questions, provide only the single, most direct answer.
            - First, identify the person whose role matches the role in the question (e.g., director, footballer).
            - If no specific role is mentioned or if the requested role is not explicitly stated but the context describes a person performing multiple key roles (like writing, producing, and starring in a film), infer that this person is the subject of the question.
            - Then, find the requested information (e.g., date of birth) for that person.
            - The answer must be a direct value from the context.
            - Do not add any additional information in parentheses.
            * **3. Insufficient information:** If the question cannot be answered because the requested information is not present for any person in the context, output exactly “Insufficient information”.

            3. Your entire response must be ONLY the JSON object. Do not add any commentary before or after it.

            Information:
            Key entities: {key_entities}
            Context: {context}

            Response (strict JSON):
            {{
            "long_answer": "Explain reasoning, clearly state if explicit or inferred, and cite which clues/documents were used.",
            "short_answer": "The single most probable answer, formatted according to the output rules."
            }}
            """
    elif type_q == "inference":
        prompt = f"""Question: {question}
        Instructions:
        1. Analyze the provided Information (Key entities, Context) to answer the Question.
        2. Your entire response MUST be a single, valid JSON object and nothing else. Do not add any commentary before or after it.
        3. The JSON object must contain four keys: "long_answer", "short_answer", "prefix", "name", and "suffix".
        - "long_answer": A detailed explanation synthesized from the information.
        - "short_answer": A single string with the direct answer, following the Rules below.
        - "prefix": Only the prefix part (or "NAN" if none).
        - "name": The core name part (must never be "NAN").
        - "suffix": Only the suffix part (or "NAN" if none).

        ---
        RULES FOR SHORT ANSWER:
        ---
        Apply these rules in the following order of priority:

        **Rule 1: Royal & Noble Figures**
        - If the question asks for a royal or noble person, construct their name precisely as follows:
        * A. START WITH: The person's full name and regnal number.
        * B. ALWAYS INCLUDE (if available): Numbered hereditary titles, noble prefixes, royal prefixes (King/Queen, Prince/Princess, Duke/Duchess, Sir/Dame, etc.).
        * C. ALWAYS EXCLUDE: Territorial/estate designations based only on land ownership.
        * D. PARENTHESES RULE: Do not include parentheses in the final string, but analyze their contents.

        **Rule 2: All Other Questions (Default Rule)**
        - If not covered by Rule 1, provide only the single, most direct answer found in the information.

        **Rule 3: Insufficient Information**
        - If the answer cannot be found in the provided information, the value of "short_answer" must be exactly "Insufficient information".
        - In this case, also set: "prefix": "NAN", "name": "Insufficient information", "suffix": "NAN".

        **Rule 4: Final Formatting Rule**
        - After generating the answer string, split it into three components:
        * Prefix: Titles before the name (or "NAN" if none).
        * Name: The personal name and regnal number (must always have a value).
        * Suffix: Titles, offices, territorial designations, or post-nominals after the name (or "NAN" if none).
        - Examples:
            - "Princess Luisa Carlotta of Naples and Sicily" -> short_answer="Princess Luisa Carlotta of Naples and Sicily"; prefix="Princess"; name="Luisa Carlotta"; suffix="of Naples and Sicily"
            - "King William IV" -> short_answer="King William IV"; prefix="King"; name="William IV"; suffix="NAN"
            - "Thomas Monck MP" -> short_answer="Thomas Monck MP"; prefix="NAN"; name="Thomas Monck"; suffix="MP"
            - "Henrietta Susanna Tracy" -> short_answer="Henrietta Susanna Tracy"; prefix="NAN"; name="Henrietta Susanna Tracy"; suffix="NAN"
            - "Insufficient information" -> short_answer="Insufficient information"; prefix="NAN"; name="Insufficient information"; suffix="NAN"

        ---
        Information:
        Key entities: {key_entities}
        Context: {context}

        Response (strict JSON only):
        {{
            "long_answer": "detailed explanation",
            "short_answer": "answer as per rules",
            "prefix": "...",
            "name": "...",
            "suffix": "..."
        }}
        """
    elif type_q == "compositional":        
        prompt = f"""Question: {question}
            Instructions:
            You are an expert intelligence analyst. Your goal is to provide the **single most probable answer** to the question. Generate a single, valid JSON object.

            ### MASTER INSTRUCTION: SYNTHESIZE ALL CONTEXT ###
            - Read ALL documents carefully.
            - Connect information across documents (e.g., if one doc names a person and another doc gives details under a synonym, link them).
            - Synonyms or alternate names MUST be treated as the same entity.

            ### CORE DIRECTIVE ###
            - You must decide if the answer is **Explicit** (directly stated) or **Inferred** (logically deduced). If explicit evidence exists, use it.  
            - If not, you are REQUIRED to infer from strong clues instead of saying "unknown", unless absolutely no clue exists.
            - Do NOT generate names of cities, regions, or countries unless they are present in the context text itself.

            ### ANSWER DERIVATION RULES ###
            1. **Explicit first**: If the answer is directly stated, use it.
            2. **Tie-breaker**: If multiple explicit answers exist, choose the most significant one (e.g., Hall of Fame > annual award, or the most recent).
            3. **Mandatory inference** (if no explicit answer):
            * (i) Long-term residence or workplace (strongest clue).
            * (ii) Professional role implied by context.
            * (iii) Nationality-based inference (fallback for “Where” questions):
                - If context only gives nationality (e.g., "Indian composer") and no other location clues, infer birthplace or death place as that country.
                - If there is evidence of migration, use the most consistent location clue instead of nationality.
            4. **Insufficient information**: If no explicit answer and no strong inference clues, output exactly "Insufficient information".

            ### RULES FOR “WHERE” QUESTIONS ###
            - Always output the **most specific location** available (city/town > state/province > country).
            - If birthplace or death place is not directly mentioned:
            * Use nationality ⇒ country as a fallback.
            * Clearly mark the answer as **inferred** in the long_answer explanation.
            

            ### OUTPUT FORMATTING (CRITICAL) ###
            - For **locations**, return the most specific unit (town/city, not country, unless only country is available).
            - For **general categories**, choose the broadest category (e.g., "cancer" not "bowel cancer").
            - For **nationalities**:
            * If "X-born Y" ⇒ answer "X".
            * If "X-Y" nationality ⇒ answer "X".
            * If "X Y" nationality ⇒ answer "Y".

            ---
            Information:
            Key entities: {key_entities}
            Context: {context}

            Response (strict JSON):
            {{
            "long_answer": "Explain reasoning, clearly state if explicit or inferred, and cite which clues/documents were used.",
            "short_answer": "The single most probable answer, formatted according to the output rules."
            }}
            """
          
    return prompt


def prompt_gen_combine_parallel_answers(question, context):
    prompt = f"""Question: {question}
        Instructions:
        1. Analyze the question and the provided information to determine the answer.
        2. Generate a response in a single, valid JSON object with two keys: "long_answer" and "short_answer".
        - long_answer: A detailed explanation synthesized from the information.
        - short_answer: A single string with the direct answer. Strictly follow these rules:
            * **1. Choice Questions (Which/Who/...):**
            - When the question asks to choose an option based on a criterion (e.g., younger, first, more recently, older, later, ...).
            - The `short_answer` must be the name of the chosen option, **extracted exactly** from the question.
            - Example: For "Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?", the short_answer is "A Flame In My Heart".

            * **2. Confirmation Questions (Are/Do/...):**
            - When the question asks to confirm a fact (e.g., same country, same nationality).
            - The `short_answer` must be **"yes"** or **"no"** (lowercase).
            - Example: For "Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?", the short_answer is "no".
        3. Your entire response must be ONLY the JSON object. Do not add any commentary before or after it.

        Information:
        Context: {context}

        Response (strict JSON):
        {{
        "long_answer": "detailed explanation",
        "short_answer": "answer as per rules"
        }}
        """
    return prompt

import json
def prompt_gen_relation(list_subquestion, type_q):
    if type_q == "inference":
        allowed_relations = {
            'father', 'mother', 'sibling', 'spouse', 'child',
            'doctoral student', 'educated at', 'field of work',
            'doctoral advisor', 'employer'
        }
    elif type_q == "compositional":
        allowed_relations = {
            'father', 'mother', 'sibling', 'spouse',
            'director', 'performer', 'composer',
            'educated at', 'founded by',
            'date of birth', 'place of birth',
            'date of death', 'place of death', 'cause of death',
            'place of burial',
            'country', 'country of origin', 'country of citizenship',
            'publication date'
        }
    elif type_q == "comparison":
        allowed_relations = set()
        data = {
            'Human': ['date of birth', 'date of death', 'date of birth and date of death (year old)', 'occupation', 'country of citizenship', 'place of birth'],
            'Film': ['publication date', 'director', 'producer', 'country of origin'],
            'Album': ['publication date', 'producer'],
            'Musical group': ['inception', 'country of origin'],
            'Song': ['publication date'],
            'General entities': ['inception', 'country'],
            'Topography': ['country']
        }
        for relations_list in data.values():
            allowed_relations.update(relations_list)

    elif type_q == "bridge_comparison":
        allowed_relations = {
            "director date of birth", "director date of death", "director country of citizenship",
            "producer date of birth", "producer date of death", "producer country of citizenship"
        }
    else:
        raise ValueError("Invalid question type. Please choose one of the following: inference, compositional, comparison, bridge_comparison.")

    system_prompt = f"""
        You are a relation extraction tool. Your task is to receive a list of sub-questions and a question type, and then identify the single relevant relation for each question from the list.

        ### Instructions
        1.  **Output Format**: Your only output must be a JSON list of strings.
        2.  **Relation Selection**: Based on the provided `type_q`, you must **only** extract relations from the following allowed set: {list(allowed_relations)}.
        3.  **One-to-One Mapping**: The number of relations in the output list **must exactly match** the number of sub-questions in the input list. The relations should be in the same order as the sub-questions.
        4.  **No Extraneous Text**: Do not include any additional text, explanations, or conversational phrases outside of the JSON output.

        ### Examples
        Example 1: If the input is:
        {{
        "list_subquestion": [
            "Who directed the movie 'Parasite'?",
            "When was the movie 'Parasite' published?",
            "What country is the movie 'Parasite' from?"
        ],
        "type_q": "compositional"
        }}
        The correct output is:
        ["director", "publication date", "country of origin"]
        ---
        Example 2: If the input is:
        {{
        "list_subquestion": [
            "Who is the father of Sir William Gore, 3Rd Baronet?",
            "Who is the father of that person?"
        ],
        "type_q": "inference"
        }}
        The correct output is:
        ["father", "father"]
        """

    user_prompt = {
        "list_subquestion": list_subquestion,
        "type_q": type_q
    }

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False, indent=2)}
    ]

    return messages


def prompt_split_question(question):
    system_prompt = """You are an AI assistant that splits a complex comparative question into a list of two simple sub-questions. Your response must ONLY be a valid JSON array of two strings. Do not include any explanation, introductory text, or markdown formatting. Your entire response must start with '[' and end with ']'."""

    user_prompt = f"""
    ### EXAMPLE 1: Complex question: "Do both directors of films The Bird of Happiness (film) and Bunco Squad share the same nationality?"

    JSON array result:
    [
    "What is the nationality of the director of 'The Bird of Happiness (film)'?",
    "What is the nationality of the director of'Bunco Squad'?"
    ]

    ### EXAMPLE 2: Complex question: "Which film has the director who is older than the other, Tiny Times 3 or Being Rose?"

    JSON array result:
    [
    "What is the date of birth of the director of 'Tiny Times 3'?",
    "What is the date of birth of the director of 'Being Rose'?"
    ]

    ### TASK: Complex question: "{question}"
    
    
    JSON array result:
    """
        
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]
         
    return messages

def get_answer_generation_system_prompt(dataset: Optional[str] = None) -> str:
    """
    Get system message for answer generation with injected rules.
    
    Args:
        dataset: Dataset context (medhop, bioasq, cwq, wiki)
        
    Returns:
        System message string
    """
    if dataset in ["medhop", "bioasq"]:
        domain_context = "You are an expert in medical and scientific knowledge."
    else:
        domain_context = "You are an expert in general knowledge."
    
    return f"""{domain_context} Your task is to:
1. Analyze the provided context carefully
2. Identify explicit facts first, then make logical inferences when needed
3. Generate accurate answers following the EXACT formatting rules provided
4. Output a valid JSON response

CRITICAL: Follow the formatting rules PRECISELY. The answer format must match the rules exactly."""


def get_answer_generation_context_synthesis_rules() -> str:
    """
    Get context synthesis rules for answer generation.
    
    Returns:
        Rules section string for prompt injection
    """
    return """
### CONTEXT SYNTHESIS RULES ###
- Read ALL provided context carefully
- Connect information across documents
- Treat synonyms/alternate names as the same entity
- Prioritize explicit evidence over inference
- Do NOT generate information (especially locations, names) not present in context
- If no explicit answer exists, infer from strong clues:
  * Long-term residence/workplace (strongest clue)
  * Professional role implied by context
  * Place of birth/nationality as fallback for location questions
- Mark inferred answers clearly in long_answer
"""


def get_answer_generation_type_specific_rules(type_q: Optional[str] = None, dataset: Optional[str] = None) -> str:
    """
    Get type-specific rules for answer generation based on question type and dataset.
    
    For MedHopQA (medhop): Returns biomedical-specific rules.
    For 2WikiMultiHopQA (wiki): Returns rules based on type_q (inference, compositional, 
    comparison, bridge_comparison).
    
    Args:
        type_q: Question type for 2WikiMultiHopQA (inference, compositional, comparison, 
                bridge_comparison). Not used for MedHopQA.
        dataset: Dataset context (medhop, wiki)
        
    Returns:
        Type-specific rules string for prompt injection
    """
    # =========================================================================
    # MEDHOP (Biomedical Domain)
    # =========================================================================
    if dataset == "medhop":
        return """
### BIOMEDICAL DOMAIN RULES ###

**Answer Format by Question Pattern:**

1. COUNT Questions ("how many", "number of"):
   - Return ONLY a number or numeric phrase
   - Examples: '10', 'ten', 'twenty-five'

2. TIME/DURATION Questions ("how long", "duration"):
   - Return concise time value with units
   - Examples: '36 hours', '3 days', '2 weeks'

3. YES/NO Questions ("is", "are", "does", "can"):
   - Return exactly 'Yes' or 'No'
   - No additional explanation in short_answer

4. ENTITY Questions ("what", "which"):
   - Return the FULL biomedical noun phrase
   - Include complete terminology, not abbreviations alone
   - Examples: 'chromosome X', 'interleukin-6', 'dopamine receptor D2'

5. ORGANISM Questions ("what organism", "what type"):
   - Return the organism type/classification
   - Examples: 'bacterium', 'fungus', 'virus', 'mammal'

6. RANGE/INTERVAL Questions ("range", "between"):
   - Return numeric range with units
   - Examples: '300-330 msec', '10-20%', '5-10 days'

7. DEFAULT (other questions):
   - Provide the most direct, specific answer from context
   - Use complete biomedical terminology
"""
    if dataset == "wiki":
        if type_q == "inference":
            return """
### INFERENCE TYPE RULES ###

This question involves family relationships or royal/noble figures.

**For ROYAL/NOBLE Figures:**
- Format answer as: prefix|name|suffix
- INCLUDE:
  * Royal prefixes: King, Queen, Prince, Princess, Duke, Duchess, Sir, Dame, Lord, Lady
  * Numbered hereditary titles: '2nd Earl of Stafford', 'III'
- EXCLUDE:
  * Territorial designations based only on land ownership
  * Parentheses in final answer

**Examples:**
- "King|William IV|"
- "Princess|Luisa Carlotta|of Naples and Sicily"
- "|Thomas Monck|MP"
- "|Henrietta Susanna Tracy|"

**For Non-Royal Figures:**
- Return the full name as found in context
"""

        elif type_q == "compositional":
            return """
### COMPOSITIONAL TYPE RULES ###

This question requires connecting facts across multiple documents.

**Core Requirements:**
- MUST synthesize information from different sources
- Treat synonyms/alternate names as the same entity
  * Example: "John Doe" = "Jean Doe" = same person

**Location Answers:**
- Return MOST SPECIFIC location available (city > state > country)
- Do NOT generate locations not present in context
- If only nationality is available, use as country fallback and mark as INFERRED

**Nationality Answers:**
- "X-born Y" (e.g., "Spanish-born Brazilian") → answer is "X" (Spanish)
- "X-Y" nationality (e.g., "Mexican-American") → answer is "X" (Mexican)
- "X Y" nationality (e.g., "German Italian") → answer is "Y" (Italian)

**General Categories:**
- Use the broadest applicable term
- Example: "cancer" instead of "bowel cancer"

**Compound Answers:**
- For multiple concepts, join with pipe '|' separator
- Example: "Presidential|Unitary|Democratic"
"""

        elif type_q in ["comparison", "bridge_comparison"]:
            return """
### COMPARISON TYPE RULES ###

This question compares two or more entities.

**Choice Questions (Which/Who is older/younger/later/earlier/first/last):**
- Return the NAME of the chosen option extracted EXACTLY from the question
- Do NOT return computed values like dates
- Example: "Which film has director born later, A Flame In My Heart or Butcher, Baker?" 
  → Answer is "A Flame In My Heart" or "Butcher, Baker", NOT a date

**Confirmation Questions (Are/Do both X and Y the same...):**
- Return 'yes' or 'no' (lowercase only)
- Do NOT include additional explanation in short_answer
- Example: "Are both directors from the same country?" → "yes" or "no"
"""

    # No specific rules for unknown dataset/type
    return ""


def get_answer_generation_output_format(type_q: Optional[str] = None, is_royal_noble: bool = False) -> str:

    if type_q == "inference" or is_royal_noble:
        return """{
    "long_answer": "Step-by-step explanation with evidence citations",
    "short_answer": "Answer following formatting rules",
    "prefix": "Royal/noble prefix or NAN",
    "name": "Core personal name",
    "suffix": "Titles after name or NAN"
}"""
    else:
        return """{
    "long_answer": "Step-by-step explanation with evidence citations",
    "short_answer": "Answer following formatting rules exactly"
}"""


def get_answer_generation_user_prompt(
    question: str,
    context_str: str,
    rules_text: str,
    output_format: str,
    key_entity: Optional[str] = None
) -> str:
    
    key_entity_text = f"\nKey Entity: {key_entity}" if key_entity else ""
    
    return f"""Task: Answer the following question based on the provided context.

        Question: {question}{key_entity_text}

        {rules_text}

        Context:
        {context_str}

        Output Format (strict JSON only):
        {output_format}

        IMPORTANT: 
        - Your entire response must be ONLY the JSON object
        - If information is not available, use "Insufficient information" as the short_answer
        - Follow the formatting rules EXACTLY"""


def get_answer_rules_prediction_system_prompt() -> str:
    
    return """You are an expert in natural language understanding and question analysis.

Your task is to analyze questions and determine the best answer formatting rules to apply.

Consider:
- Question patterns (yes/no, comparison, choice, etc.)
- Entity types being asked about (person, place, biomedical, etc.)
- Special cases (royal/noble figures, compound answers, etc.)"""


def get_answer_rules_prediction_user_prompt(
    question: str,
    rules_reference: str,
    dataset: Optional[str] = None,
    type_q: Optional[str] = None
) -> str:
    """
    Build user prompt for answer rules prediction.
    
    Args:
        question: The question to analyze
        rules_reference: JSON string of available rules
        dataset: Dataset context (medhop, bioasq, cwq, wiki)
        type_q: Question type hint (inference, compositional, comparison, bridge_comparison)
        
    Returns:
        Complete user prompt string
    """
    context_info = ""
    if dataset:
        context_info += f"\nDataset: {dataset}"
        if dataset in ["medhop", "bioasq"]:
            context_info += " (biomedical/scientific domain)"
        elif dataset == "wiki":
            context_info += " (general knowledge, Wikipedia-based)"
        elif dataset == "cwq":
            context_info += " (complex web questions)"
    
    if type_q:
        context_info += f"\nQuestion Type Hint: {type_q}"
        if type_q == "inference":
            context_info += " (family relationships, royal/noble figures)"
        elif type_q == "compositional":
            context_info += " (multi-hop facts about entities)"
        elif type_q in ["comparison", "bridge_comparison"]:
            context_info += " (comparing two entities)"

    return f"""Analyze this question and determine the appropriate answer formatting rules.

Question: "{question}"
{context_info}

Available Rules Reference:
{rules_reference}

Task: Analyze the question and output a JSON object with:
1. "question_type": The primary type of question (from the rules reference keys)
2. "primary_rule": The main rule name that should be applied
3. "secondary_rules": Array of additional rule names that may also apply
4. "formatting_instructions": Specific formatting instructions for this question
5. "examples": 2-3 example answers that would follow the rules
6. "reasoning": Brief explanation of why these rules were chosen

Output ONLY valid JSON, no additional text."""


# ============================================================================
# ANSWER SELECTION PROMPTS - Extracted from handle_multi_choice.py
# ============================================================================

def get_answer_selection_user_prompt(
    question: str,
    answers: List[Dict[str, Any]]
) -> str:
    """
    Build user prompt for answer selection from multiple candidates.
    
    Optimized for o3-mini: minimalist and constraint-focused.
    
    Args:
        question: The question being answered
        answers: List of answer dicts with 'short_answer' and 'long_answer'
        
    Returns:
        Complete user prompt string
    """
    # Format answers for comparison
    formatted_answers = "\n\n".join([
        f"[{idx + 1}] Short: {ans.get('short_answer', 'N/A')}\n    Long: {ans.get('long_answer', 'N/A')}"
        for idx, ans in enumerate(answers)
    ])
    
    return f"""Goal: Select the single best answer to the question from {len(answers)} candidates.

Constraints:
- Output valid JSON only
- Choose exactly one answer (1-{len(answers)})
- Selection criteria (in priority order):
  1. Accuracy: Answer correctly addresses the question
  2. Specificity: Answer is appropriately detailed/specific
  3. Evidence quality: Reasoning in long_answer is well-supported
  4. Consistency: Answer is internally consistent
- For identical/equivalent answers, prefer the one with better reasoning
- For conflicting answers, choose based on evidence strength

Question: {question}

Candidates:
{formatted_answers}

Required JSON format:
{{
    "answer_index": <1 to {len(answers)}>,
    "reasoning": "Brief explanation of why this answer is best",
    "confidence": "high|medium|low"
}}"""


def get_parallel_answers_combination_prompt(question: str, context: str) -> str:
    """
    Get prompt for combining parallel answers from multiple sources.
    
    Used in generate_combine_parallel_answers().
    
    Args:
        question: The question being answered
        context: Combined context from multiple sources
        
    Returns:
        Prompt string
    """
    return f"""Question: {question}
Instructions:
1. Analyze the question and the provided information to determine the answer.
2. Generate a response in a single, valid JSON object with two keys: "long_answer" and "short_answer".
- long_answer: A detailed explanation synthesized from the information.
- short_answer: A single string with the direct answer. Strictly follow these rules:
    * **1. Choice Questions (Which/Who/...):**
    - When the question asks to choose an option based on a criterion (e.g., younger, first, more recently, older, later, ...).
    - The `short_answer` must be the name of the chosen option, **extracted exactly** from the question.
    - Example: For "Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?", the short_answer is "A Flame In My Heart".

    * **2. Confirmation Questions (Are/Do/...):**
    - When the question asks to confirm a fact (e.g., same country, same nationality).
    - The `short_answer` must be **"yes"** or **"no"** (lowercase).
    - Example: For "Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?", the short_answer is "no".
3. Your entire response must be ONLY the JSON object. Do not add any commentary before or after it.

Information:
Context: {context}

Response (strict JSON):
{{
"long_answer": "detailed explanation",
"short_answer": "answer as per rules"
}}
"""


def get_multiple_choice_selection_prompt(question: str, context: str) -> str:
    """
    Get prompt for selecting answer from multiple choice options.
    
    Used in gen_ans_multiple_choice().
    
    Args:
        question: The question being answered
        context: Context containing multiple choice options
        
    Returns:
        Prompt string
    """
    return f"""Question: {question}
Instructions:
- Please base your answer ONLY on the context provided.
- Please only choose the given answers (without additional explanations). For example: "A", "B", "C", "D", "E", "F", "G".
- Do not include anything else aside from your answer.

Context:
{context}
Please provide your short answer to the question based on the context above."""

COMPREHENSIVE_VALIDATION_RULES = {
    "question_type_detection": {
        "description": "Identify and match question type patterns",
        "categories": {
            "count": {
                "keywords": ["how many", "how much", "number of", "total"],
                "format": "ONLY a number or numeric phrase (e.g., '10', 'ten')",
                "rule": "No extra text allowed"
            },
            "time_duration": {
                "keywords": ["how long", "duration", "how many hours", "how many days"],
                "format": "Concise time value (e.g., '36 hours', '3 days')",
                "rule": "Must be time-specific"
            },
            "yes_no": {
                "keywords": ["is ", "are ", "do ", "does ", "can ", "could ", "will ", "would ", "should "],
                "format": "Exactly 'Yes' or 'No'",
                "rule": "Binary answer only"
            },
            "entity_biomedical": {
                "keywords": ["what", "who", "which"],
                "domain": "biomedical/scientific",
                "format": "FULL noun phrase (e.g., 'chromosome X', 'interleukin-6', 'dopamine receptor D2')",
                "rule": "Must be complete biomedical terminology"
            },
            "entity_general": {
                "keywords": ["what", "who", "which"],
                "domain": "general knowledge/people/films/places",
                "format": "Proper names are VALID (e.g., 'Manuel Romero', 'Paris')",
                "rule": "Context determines biomedical vs general"
            },
            "organism": {
                "keywords": ["what organism", "what type of organism"],
                "format": "Organism type (e.g., 'bacterium', 'fungus')",
                "rule": "Must be organism classification"
            },
            "range_interval": {
                "keywords": ["range", "between", "from", "to"],
                "format": "Numeric range (e.g., '300-330 msec', '10-20%')",
                "rule": "Must be interval format"
            },
            "comparison": {
                "keywords": ["older", "younger", "later", "earlier", "which", "who has"],
                "format": "NAME of chosen option extracted EXACTLY from question",
                "rule": "Return option name, NOT computed value (not a date for older/younger)"
            },
            "confirmation": {
                "keywords": ["are ", "is ", "same", "different", "both"],
                "format": "'yes' or 'no' (lowercase)",
                "rule": "Binary confirmation only"
            },
            "inference_royal_noble": {
                "keywords": ["father", "mother", "son", "daughter", "king", "queen", "prince", "princess"],
                "format": "prefix|name|suffix (e.g., 'King|William IV|')",
                "rule": "Include royal prefixes and hereditary titles, EXCLUDE territorial designations"
            },
            "compositional": {
                "keywords": ["where", "nationality", "education", "occupation"],
                "format": "Entity-specific (most specific location: city > state > country)",
                "rule": "Connect facts across multiple documents"
            }
        }
    },
    "medhop_biomedical_rules": {
        "description": "Rules specific to MedHop/BioASQ biomedical domain",
        "count_questions": {
            "rule": "Answer must be ONLY a number or numeric phrase",
            "examples": ["10", "ten", "twenty-five"],
            "violations": ["extra text", "non-numeric content"]
        },
        "time_duration_questions": {
            "rule": "Answer must be concise time value",
            "examples": ["36 hours", "3 days", "2 weeks"],
            "violations": ["imprecise duration", "narrative time references"]
        },
        "yes_no_questions": {
            "rule": "Answer must be exactly 'Yes' or 'No'",
            "violations": ["yes/no with explanation", "maybe", "uncertain"]
        },
        "entity_questions": {
            "rule": "Return FULL noun phrase for biomedical/scientific entities",
            "examples": ["chromosome X", "interleukin-6", "dopamine receptor D2"],
            "violations": ["partial names", "abbreviations without full form"]
        },
        "organism_questions": {
            "rule": "Return organism type/classification",
            "examples": ["bacterium", "fungus", "virus"],
            "violations": ["specific organism name instead of type"]
        },
        "range_questions": {
            "rule": "Return numeric range with units",
            "examples": ["300-330 msec", "10-20%"],
            "violations": ["single value instead of range"]
        }
    },
    "wiki_general_knowledge_rules": {
        "description": "Rules for Wikipedia/general knowledge domain",
        "comparison_questions": {
            "rule": "Answer must be the NAME of chosen option extracted EXACTLY from question",
            "examples": [
                "Question: 'Which film has director born later, A Flame In My Heart or Butcher, Baker?'",
                "Answer: 'A Flame In My Heart' (NOT a date)"
            ],
            "violations": ["computed value", "date instead of option name", "paraphrased option"]
        },
        "confirmation_questions": {
            "rule": "Answer must be 'yes' or 'no' (lowercase)",
            "examples": ["yes", "no"],
            "violations": ["Yes/No (capitalized)", "explanation after yes/no"]
        },
        "inference_royal_noble": {
            "rule": "Format: prefix|name|suffix with mandatory royal prefixes",
            "prefixes_required": ["King", "Queen", "Prince", "Princess", "Duke", "Duchess", "Sir", "Dame", "Lord", "Lady"],
            "include": ["numbered hereditary titles: '2nd Earl of Stafford'"],
            "exclude": ["territorial designations based only on land ownership"],
            "examples": ["King|William IV|", "Princess|Luisa Carlotta|of Naples and Sicily"],
            "violations": ["missing royal prefix", "parentheses in answer", "territorial designations"]
        },
        "compound_answers": {
            "rule": "Deconstruct multiple concepts, join with pipe '|' separator",
            "examples": ["Presidential|Unitary|Democratic", "Drawing|Painting|Photography"],
            "violations": ["unseparated concepts", "single compound noun"]
        }
    },
    "location_rules": {
        "description": "Rules for location/geography answers",
        "specificity": "Use MOST SPECIFIC (city > state > country)",
        "generation": "Do NOT generate location names not in context",
        "fallback": {
            "rule": "For birthplace/death without explicit mention: nationality → country as fallback",
            "must_mark": "Clearly mark as INFERRED if using nationality fallback"
        },
        "violations": ["too general location", "hallucinated place", "incorrect specificity"]
    },
    "nationality_rules": {
        "description": "Rules for nationality/citizenship answers",
        "patterns": {
            "x_born_y": {"pattern": "X-born Y", "answer": "X"},
            "x_y_nationality": {"pattern": "X-Y nationality", "answer": "X"},
            "x_y_space": {"pattern": "X Y nationality", "answer": "Y"}
        }
    },
    "content_validation": {
        "description": "General content validation rules",
        "requirements": [
            "No hallucinated entities/locations",
            "Synonyms/alternate names treated as same entity",
            "Evidence properly cited in long_answer",
            "Explicit evidence prioritized over inference",
            "'Insufficient information' is VALID answer when context truly lacks information"
        ],
        "long_answer_requirements": {
            "flexibility": "Accept any reasonable explanation connecting evidence to answer",
            "optional": ["citing document IDs", "step-by-step format", "rigid structure"],
            "priority": "Check that long_answer is consistent with short_answer and contains reasoning",
            "strict": False
        }
    },
    "insufficient_information": {
        "description": "Rules for 'Insufficient information' answers",
        "when_valid": "Context truly lacks needed information",
        "short_answer": "Insufficient information",
        "long_answer": "Should explain why information is unavailable, but be flexible with format",
        "content_check": "SKIP strict content checks for this answer type"
    }
}


def get_rules_for_question(question: str, dataset: Optional[str] = None, type_q: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all applicable rules for a question without using if-else logic.
    
    This function analyzes the question and returns matching rules from COMPREHENSIVE_VALIDATION_RULES
    based on keyword matching and patterns, allowing for generalization across question types.
    
    Args:
        question: The question text to analyze
        dataset: Optional dataset context (medhop, bioasq, cwq, wiki)
        type_q: Optional question type hint (inference, compositional, comparison, bridge_comparison)
        
    Returns:
        Dict containing all relevant rules, question detection, and formatting requirements
    """
    question_lower = question.lower()
    applicable_rules = {}
    
    # Detect question type by keyword matching (no if-else)
    question_type_detection = COMPREHENSIVE_VALIDATION_RULES["question_type_detection"]
    matched_types = []
    
    for qtype, qtype_info in question_type_detection["categories"].items():
        keywords = qtype_info.get("keywords", [])
        if any(kw in question_lower for kw in keywords):
            matched_types.append({
                "type": qtype,
                "format": qtype_info.get("format", ""),
                "rule": qtype_info.get("rule", ""),
                "keywords_matched": [kw for kw in keywords if kw in question_lower]
            })
    
    applicable_rules["detected_question_types"] = matched_types
    
    # Apply domain-specific rules based on dataset
    if dataset in ["medhop", "bioasq"]:
        applicable_rules["domain"] = "biomedical"
        applicable_rules["domain_rules"] = COMPREHENSIVE_VALIDATION_RULES["medhop_biomedical_rules"]
    elif dataset in ["cwq", "wiki"]:
        applicable_rules["domain"] = "general_knowledge"
        applicable_rules["domain_rules"] = COMPREHENSIVE_VALIDATION_RULES["wiki_general_knowledge_rules"]
    else:
        applicable_rules["domain"] = "unknown"
    
    # Add universal rules
    applicable_rules["location_rules"] = COMPREHENSIVE_VALIDATION_RULES["location_rules"]
    applicable_rules["nationality_rules"] = COMPREHENSIVE_VALIDATION_RULES["nationality_rules"]
    applicable_rules["content_validation"] = COMPREHENSIVE_VALIDATION_RULES["content_validation"]
    applicable_rules["insufficient_information"] = COMPREHENSIVE_VALIDATION_RULES["insufficient_information"]
    
    # Add type-specific hint if provided
    if type_q:
        applicable_rules["type_q_hint"] = type_q
    
    return applicable_rules


def get_prompt_for_answer_validation(question: str, answer: dict, dataset: str = "", type_q: str = "") -> List[Dict[str, str]]:
    """
    Generate validation prompt for checking if answer follows normalization rules.
    Uses comprehensive validation checklist covering all question types and domains.
    NO IF-ELSE LOGIC - All rules embedded in prompt for LLM to detect and apply.
    
    Args:
        question: The question being answered
        answer: Dictionary with 'short_answer' and 'long_answer' keys
        dataset: Dataset name (optional, for context only - not used for conditional logic)
        type_q: Question type (optional, for context only - not used for conditional logic)
    
    Returns:
        List of message dictionaries for GPT API
    
    Example:
        >>> messages = get_prompt_for_answer_validation(
        ...     question="How many genes are involved?",
        ...     answer={"short_answer": "5", "long_answer": "Based on the evidence..."}
        ... )
        >>> response = gpt_request(messages, model="gpt-4o-mini")
    """
    system_message = """You are an expert answer validator for multi-hop question answering systems. Your task is to validate answers against comprehensive normalization rules covering biomedical, general knowledge, and specialized question types."""
    
    user_prompt = f"""Task: Validate if this answer follows all normalization rules. Think step-by-step.

Question: {question}
Short Answer: {answer.get('short_answer', '')}
Long Answer: {answer.get('long_answer', '')}

COMPREHENSIVE VALIDATION CHECKLIST:

1. QUESTION TYPE DETECTION:
   - Identify question type: count ("how many"), time/duration, yes/no, entity (Who/Which/What), organism, comparison, inference, compositional
   - Verify answer format matches detected type

2. BIOMEDICAL RULES:
   a) Count Questions:
      - Answer must be ONLY a number or numeric phrase (e.g., "10", "ten")
      - No extra text allowed
   
   b) Time/Duration Questions:
      - Answer must be concise time value (e.g., "36 hours", "3 days")
   
   c) Yes/No Questions:
      - Answer must be exactly "Yes" or "No"
   
   d) Entity Questions (Who/Which/What):
      - For BIOMEDICAL/SCIENTIFIC entities: Must return FULL noun phrase (e.g., "chromosome X", "interleukin-6", "dopamine receptor D2")
      - For GENERAL KNOWLEDGE (people/films/places): Proper names are VALID (e.g., "Manuel Romero", "Paris")
      - Check context to determine if question is biomedical or general knowledge
   
   e) Organism Questions:
      - Return organism type (e.g., "bacterium", "fungus")
   
   f) Range/Interval Questions:
      - Return numeric range (e.g., "300-330 msec", "10-20%")
   
   g) Lists:
      - If none of above apply, provide concise comma-separated items

3. WIKI/GENERAL KNOWLEDGE RULES:
   a) Comparison Questions (Which/Who...older/younger/later/earlier):
      - Answer must be the NAME of chosen option extracted EXACTLY from question
      - Example: For "Which film has director born later, A or B?", answer is "A" NOT a date
   
   b) Confirmation Questions (Are/Do...same):
      - Answer must be "yes" or "no" (lowercase)
   
   c) Inference Questions (Royal/Noble Figures):
      - Royal prefix required if applicable: King/Queen, Prince/Princess, Duke/Duchess, etc.
      - Include numbered hereditary titles: "2nd Earl of Stafford"
      - Include honorifics: Sir/Dame/Lord/Lady
      - EXCLUDE territorial designations based only on land ownership
      - NO parentheses in final answer (but analyze them for context)
      - Format for royal/noble: "prefix|name|suffix" (e.g., "King|William IV|")
   
   d) Compound Answers:
      - Multiple concepts MUST be deconstructed
      - Join with pipe "|" separator
      - Example: "Presidential unitary democratic republic" → "Presidential|Unitary|Democratic"

4. LOCATION RULES:
   - Always use MOST SPECIFIC location (city > state > country)
   - Do NOT generate location names not in context
   - For birthplace/death without explicit mention: use nationality → country as fallback
   - Clearly mark as INFERRED if using nationality fallback

5. NATIONALITY RULES:
   - "X-born Y" → answer "X"
   - "X-Y nationality" → answer "X"
   - "X Y nationality" → answer "Y"

6. CONTENT VALIDATION:
   - No hallucinated entities/locations
   - Synonyms/alternate names treated as same entity
   - Evidence properly cited in long_answer
   - Explicit evidence prioritized over inference
   - "Insufficient information" is a VALID answer when the context truly lacks needed information

7. SPECIAL CASE - "Insufficient information":
   - Short answer = "Insufficient information" → SKIP strict content checks
   - Long_answer should explain why information is unavailable, but be flexible with format

8. LONG ANSWER VALIDATION (FLEXIBLE):
   - Long answer should generally support the short answer with reasoning
   - Accept any reasonable explanation that connects evidence to the answer
   - Citing document IDs (like Q6752853) or specific evidence sources is good but not required
   - Step-by-step explanations are encouraged but not mandatory
   - PRIORITY: Check only that long_answer is consistent with short_answer and contains some reasoning
   - Do NOT enforce rigid formatting requirements on the long answer

Output Format (strict JSON):
{{
  "is_valid": true/false,
  "violations": ["list of specific rule violations found with rule category, empty if valid"],
  "additional_rules": "Specific corrective instructions to inject into generator prompt (only if is_valid=false, otherwise empty string)"
}}"""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages
