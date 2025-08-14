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
    elif dataset == "bioasq":
        prompt = prompt_gen_answer_bioasq_step_3b(question, key_entities, relevant_info)
    elif dataset == "cwq":
        prompt = prompt_gen_answer_cwq_step_3b(question, key_entities, relevant_info)
    elif dataset == "wiki":
        prompt = prompt_gen_answer_wiki_step_3b(question, key_entities, relevant_info, type_q=type_q)

    return prompt

def _load_search_keyword_prompt(question, dataset=None, type_q=None):
    if dataset == 'medhop':
        messages = [
            {
        "role": "system",
        "content": """You are an expert in generating ONLY 1 concise and effective Google search query to help answer a complex biomedical question.
    Your goal is to provide a single, keyword-based query string that best targets the core information needed from the original question.

    Key principles for the single query:
    1. The query should be a distillation of the original complex question, focusing on essential keywords.
    2. The query MUST be concise, using specific keywords INCLUDING IMPORTANT BIOMEDICAL KEYWORD, nouns, and technical terms critical to answering the question.

    The final output MUST be a JSON array containing a single object. This object must contain a single key "search_keyword", whose value is the Google search query string.
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
        {{ "search_keyword": "Smith-Fineman-Myers syndrome mutated gene helicase domain" }}
    ]
    ---
    **Example 2:**

    Question: "Which lactate dehydrogenase tetramer is most common in the primary organ of the nervous system?"

    JSON desired output format:
    [
        {{ "search_keyword": "lactate dehydrogenase tetramer most common primary organ nervous system" }}
    ]
    ---
    **Example 3:**

    Question: "Which physician, along with Edvard Ehlers, is the Ehlers Danlos syndromes named after?"

    JSON desired output format:
    [
        {{ "search_keyword": "Ehlers Danlos syndromes named after along with Edvard Ehlers" }}
    ]
    ---
    **Example 4:**

    Question: "Which benign, autosomal recessive hyperbilirubinemia syndrome is caused by a mutation in the UGT1A1 gene on chromosome 2?"

    JSON desired output format:
    [
        {{ "search_keyword": "benign autosomal recessive hyperbilirubinemia UGT1A1 gene mutation chromosome 2" }}
    ]
    ---
    **Example 5:**

    Question: "Which recognized type of osteochondritis is characterized by uneven vertebral growth and kyphosis that typically develops during adolescence?"

    JSON desired output format:
    [
        {{ "search_keyword": "osteochondritis type uneven vertebral growth kyphosis adolescence" }}
    ]
    ---

    Now, process the following:

    Question: "{question}"

    Task: Generate ONLY 1 effective Google search query that distills the core of the given question.

    Return a JSON array containing a single object. This object must contain a single key "search_keyword", whose value is the Google search query string. For example:
    [
        {{ "search_keyword": "..." }}
    ]
    """
            }
        ]
    elif dataset == 'multihoprag':
        messages = [
            {
                "role": "system",
                "content": """You are an expert in generating ONLY 1 concise and effective Google search query to help answer a complex question.
        Your goal is to provide a single, keyword-based query string that best targets the core information needed from the original question.

        Key principles for the single query:
        1. The query should be a distillation of the original complex question, focusing on essential keywords.
        2. The query MUST be concise, using specific keywords, proper nouns, and technical terms critical to answering the question.

        The final output MUST be a JSON array containing a single object. This object must contain a single key "search_keyword", whose value is the Google search query string.
        """
            },
            {
                "role": "user",
                "content": f"""Here are 6 examples demonstrating the desired JSON output format, covering various domains such as technology, entertainment, sports, science, business, and health.

        ---
        **Example 1 (Technology):**

        Question: "What was the name of the first commercially successful video game console that used interchangeable ROM cartridges, released by Fairchild Camera and Instrument?"

        JSON desired output format:
        [
            {{ "search_keyword": "first commercially successful video game console interchangeable ROM cartridges Fairchild" }}
        ]
        ---
        **Example 2 (Entertainment):**

        Question: "Which 2019 film directed by Bong Joon-ho won the Academy Award for Best Picture, becoming the first non-English language film to do so?"

        JSON desired output format:
        [
            {{ "search_keyword": "Bong Joon-ho 2019 film best picture first non-english" }}
        ]
        ---
        **Example 3 (Sports):**

        Question: "In the 1998 NBA Finals, which player made the famous 'Last Shot' over Bryon Russell to clinch the championship for the Chicago Bulls?"

        JSON desired output format:
        [
            {{ "search_keyword": "1998 NBA Finals player 'Last Shot' over Bryon Russell Chicago Bulls" }}
        ]
        ---
        **Example 4 (Science):**

        Question: "What is the name of the theoretical boundary around a black hole beyond which events cannot affect an outside observer, often called the 'point of no return'?"

        JSON desired output format:
        [
            {{ "search_keyword": "theoretical boundary black hole point of no return" }}
        ]
        ---
        **Example 5 (Business):**

        Question: "What is the economic principle that describes how, in a market with information asymmetry, the presence of low-quality goods can drive high-quality goods out of the market?"

        JSON desired output format:
        [
            {{ "search_keyword": "economic principle information asymmetry low-quality drives out high-quality" }}
        ]
        ---
        **Example 6 (Health):**

        Question: "What is the name of the vaccine that protects against measles, mumps, and rubella, typically administered in two doses during childhood?"

        JSON desired output format:
        [
            {{ "search_keyword": "vaccine name for measles mumps rubella" }}
        ]
        ---

        Now, process the following:

        Question: "{question}"

        Task: Generate ONLY 1 effective Google search query that distills the core of the given question.

        Return a JSON array containing a single object. This object must contain a single key "search_keyword", whose value is the Google search query string. For example:
        [
            {{ "search_keyword": "..." }}
        ]
        """
            }
        ]
    elif dataset == 'cwq':
        messages = [
           {
       "role": "system",
       "content": """You are an expert in generating ONLY 1 concise and effective Google search query to help answer a complex question.
        Your goal is to provide a single, keyword-based query string that best targets the core information needed from the original question.


        Key principles for the single query:
        1. The query should be a distillation of the original complex question, focusing on essential keywords, proper names, and constraints.
        2. The query MUST be concise, using specific nouns and terms critical to answering the question.


        The final output MUST be a JSON array containing a single object. This object must contain a single key "search_keyword", whose value is the Google search query string.
        """
                },
                {
            "role": "user",
            "content": f"""Here are 5 examples demonstrating the desired JSON output format, which is a JSON array containing a single object with a "search_keyword" field holding the Google search query string.


        ---
        **Example 1:**


        Question: "Who held his governmental position the latest and won the 2007 Nobel Peace Prize?"


        JSON desired output format:
        [
            {{ "search_keyword": "2007 Nobel Peace Prize winner latest governmental position" }}
        ]
        ---
        **Example 2:**


        Question (conjunction): "The city/town of Lyon is in what country containing Tomtom Western Europe?"


        JSON desired output format:
        [
            {{ "search_keyword": "country in Tomtom Western Europe that contains Lyon" }}
        ]
        ---
        **Example 3:**


        Question: "Beijing is the capital of what country in the Vietnam War?"


        JSON desired output format:
        [
            {{ "search_keyword": "country in Vietnam War with capital Beijing" }}
        ]
        ---
        **Example 4:**


        Question: "What catholic influenced sir authur conan doyle?"


        JSON desired output format:
        [
            {{ "search_keyword": "catholic person who influenced Arthur Conan Doyle" }}
        ]
        ---
        **Example 5:**


        Question: "Which country has a standing army of less than 12000 and includes the Panama Canal?"


        JSON desired output format:
        [
            {{ "search_keyword": "country with Panama Canal and army less than 12000" }}
        ]
        ---


        Now, process the following:


        Question: "{question}"


        Task: Generate ONLY 1 effective Google search query that distills the core of the given question.


        Return a JSON array containing a single object. This object must contain a single key "search_keyword", whose value is the Google search query string. For example:
        [
            {{ "search_keyword": "..." }}
        ]
        """
                }
            ]
    elif dataset == 'wiki':
        if type_q == "comparison":
            # messages = [
            #     {
            #         "role": "system",
            #         "content": """You are an expert in generating ONLY 1 concise and effective Google search query to help answer a complex question.
            #         Your goal is to provide a single, keyword-based query string that best targets the core information needed from the original question.

            #         Key principles for the single query:
            #         1. The query should be a distillation of the original complex question, focusing on essential keywords, proper names, and the core relationship being asked about.
            #         2. The query MUST be concise, using specific nouns and terms critical to answering the question, and should omit conversational parts like "Who is the".
            #         3. The key entity is the main subject (person, place, thing, or concept) the question is about.

            #         The final output MUST be a JSON array containing a single object. This object must contain two keys:
            #         - "search_keyword": The generated Google search query string.
            #         - "key_entity": The primary subject extracted from the question.
            #         """
            #     },
            #     {
            #         "role": "user",
            #         "content": f"""Here are 5 examples demonstrating the desired JSON output format, which is a JSON array containing a single object with a "search_keyword" field holding the Google search query string.

            #         ---
            #         **Example 1:**

            #         Question: "Who directed the (1957 Film) 'Just Once A Great Lady'?"

            #         JSON desired output format:
            #         [
            #             {{
            #                 "search_keyword": "director 'Just Once A Great Lady (1957 Film)'",
            #                 "key_entity": "Just Once A Great Lady (1957 Film)"
            #             }}
            #         ]
            #         ---
            #         **Example 2:**

            #         Question: "In which country is Carnival Of Souls located??"

            #         JSON desired output format:
            #         [
            #             {{ 
            #                 "search_keyword": "Carnival Of Souls country",
            #                 "key_entity": "Carnival Of Souls"
            #             }}
            #         ]
            #         ---
            #         **Example 3:**

            #         Question: "When was the film Engal Aasan released?"

            #         JSON desired output format:
            #         [
            #             {{
            #                 "search_keyword": "Engal Aasan release date",
            #                 "key_entity": "Engal Aasan"
            #             }}
            #         ]
            #         ---
            #         **Example 4:**

            #         Question: "When was Michelle Tong (Writer) born?"

            #         JSON desired output format:
            #         [
            #             {{
            #                 "search_keyword": "Michelle Tong (Writer) birth date",
            #                 "key_entity": "Michelle Tong (Writer)"
            #             }}
            #         ]
            #         ---
            #         **Example 5:**

            #         Question: "Who is William C. White's father?"

            #         JSON desired output format:
            #         [
            #             {{
            #                 "search_keyword": "William C. White father",
            #                 "key_entity": "William C. White"
            #             }}
            #         ]
            #         ---
            #         Now, process the following:
            #         Question: "{question}"
            #         Task: Generate ONLY 1 effective Google search query that distills the core of the given question.
            #         Return a JSON array containing a single object with the keys "search_keyword" and "key_entity", following the format of the examples above.
            #         [
            #             {{ 
            #                 "search_keyword": "...",
            #                 "key_entity": "..." 
            #             }}
            #         ]
            #         """
            #     }
            # ]
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert in generating ONLY 1 concise and effective Google search query to help answer a complex question.
                    Your goal is to provide a single, keyword-based query string that best targets the core information needed from the original question.

                    Key principles for the single query:
                    1. The query should be a distillation of the original complex question, focusing on essential keywords, proper names, and the core relationship being asked about.
                    2. The query MUST be concise, using specific nouns and terms critical to answering the question, and should omit conversational parts like "Who is the".
                    3. The key entity is the main subject (person, place, thing, or concept) the question is about.
                    4. **Adapt relation keywords based on the key entity's type.** This is crucial for avoiding ambiguity. For example:
                    - For a **Person** (Human), use specific relations like: `country of citizenship`, `date of birth`, `occupation`,...
                    - For a **Creative Work** (Film, Album, Song), use relations like: `country of origin`, `publication date`, `director`, `producer`,...
                    - For a **Location or Organization** (Building, Museum, University, Mountain), use relations like: `country`, `inception`,...

                    The final output MUST be a JSON array containing a single object. This object must contain two keys:
                    - "search_keyword": The generated Google search query string.
                    - "key_entity": The primary subject extracted from the question.
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
                            "key_entity": "Just Once A Great Lady (1957 Film)"
                        }}
                    ]
                    ---
                    **Example 2:**

                    Question: "In which country is the film Carnival Of Souls from?"

                    JSON desired output format:
                    [
                        {{ 
                            "search_keyword": "'Carnival Of Souls' country of origin",
                            "key_entity": "Carnival Of Souls"
                        }}
                    ]
                    ---
                    **Example 3:**

                    Question: "When was the film Engal Aasan released?"

                    JSON desired output format:
                    [
                        {{
                            "search_keyword": "'Engal Aasan' publication date",
                            "key_entity": "Engal Aasan"
                        }}
                    ]
                    ---
                    **Example 4:**

                    Question: "When was Michelle Tong (Writer) born?"

                    JSON desired output format:
                    [
                        {{
                            "search_keyword": "'Michelle Tong (Writer)' date of birth",
                            "key_entity": "Michelle Tong (Writer)"
                        }}
                    ]
                    ---
                    **Example 5:**

                    Question: "What is the nationality of the writer William C. White?"

                    JSON desired output format:
                    [
                        {{
                            "search_keyword": "'William C. White' country of citizenship",
                            "key_entity": "William C. White"
                        }}
                    ]
                    ---
                    Now, process the following:
                    Question: "{question}"
                    Task: Generate ONLY 1 effective Google search query that distills the core of the given question.
                    Return a JSON array containing a single object with the keys "search_keyword" and "key_entity", following the format of the examples above.
                    [
                        {{ 
                            "search_keyword": "...",
                            "key_entity": "..." 
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
                "subquestion": "Which enzyme can inactivate antithrombin?",
                "search_keyword": "enzyme inactivate antithrombin"
            }},
            {{
                "subquestion": "Which gene encode that enzyme?",
                "search_keyword": "gene encode enzyme inactivate antithrombin"
            }},
            {{
                "subquestion": "Which human chromosome contains that gene?",
                "search_keyword": "human chromosome gene encode enzyme inactivate antithrombin"
            }}
        ]

        ---
        **Example 2:**
        "Which lactate dehydrogenase tetramer is most common in the primary organ of the nervous system?"

        JSON desired output format:
        [
            {{
                "subquestion": "What is the primary organ of the human nervous system?",
                "search_keyword": "primary organ human nervous system"
            }},
            {{
                "subquestion": "Which specific lactate dehydrogenase tetramer is most commonly found in that organ?",
                "search_keyword": "lactate dehydrogenase tetramer most common primary organ nervous system"
            }}
        ]

        ---
        **Example 3:**
        "Which gene mutation is responsible for the syndrome characterized by flattened and deformed vertebrae, developmental delay, and lens coloboma?"

        JSON desired output format:
        [
            {{
                "subquestion": "Which syndrome is characterized by flattened and deformed vertebrae, developmental delay, and lens coloboma?",
                "search_keyword": "syndrome flattened deformed vertebrae developmental delay lens coloboma"
            }},
            {{
                "subquestion": "Which gene mutation is responsible for that syndrome?",
                "search_keyword": "gene mutation syndrome flattened deformed vertebrae developmental delay lens coloboma"
            }}
        ]

        ---
        **Example 4:**
        "Which London suburb was home to the pioneering institutions that influenced John Langdon Down's work at Earlswood Asylum?"

        JSON desired output format:
        [
            {{
                "subquestion": "Which pioneering institutions influenced John Langdon Down's work at Earlswood Asylum?",
                "search_keyword": "pioneering institutions John Langdon Down Earlswood Asylum"
            }},
            {{
                "subquestion": "In which London suburb was that pioneering institution located?",
                "search_keyword": "London location pioneering institution John Langdon Down Earlswood Asylum"
            }}
        ]

        ---
        **Example 5:**
        "Which gene mutation is primarily responsible for a syndromic microphthalmia that includes nasal abnormalities?"

        JSON desired output format:
        [
            {{
                "subquestion": "What are the syndromic forms of microphthalmia that present with nasal abnormalities?",
                "search_keyword": "syndromic microphthalmia nasal abnormalities"
            }},
            {{
                "subquestion": "Which gene mutation is most commonly identified as the primary cause for that disease?",
                "search_keyword": "gene mutation primary cause syndromic microphthalmia nasal abnormalities"
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
    elif dataset == 'multihoprag':
        messages = [
            {
                "role": "system",
                "content": f"""You are a highly specialized AI assistant, an expert at decomposing complex questions into a sequence of simpler, interconnected natural language sub-questions. Your primary goal is to create a logical, step-by-step plan that faithfully represents the reasoning process to answer the main question.

        **Core Instructions for Decomposition:**

        0.  **Faithful Reasoning via Decomposition**: Your primary task is to externalize the reasoning process by breaking the `Original Complex Question` into a series of sub-questions. This decomposition itself forms the transparent and explainable path to the solution.
        1.  **Step-by-Step Plan**: Generate a sequence of sub-questions.
        2.  **Self-Contained and Searchable Sub-questions**: Each sub-question MUST be formulated so it is **self-contained and understandable in isolation**. This means it must include all necessary context from the original question or previous steps to be answerable by a standard search engine query *without needing to refer back to the original complex question or other sub-questions implicitly*. Sub-questions must be specific and unambiguous. For example, avoid pronouns or references that don't have explicit antecedents within the sub-question itself.
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
        Here are 6 examples demonstrating the desired JSON output format and decomposition logic across various domains:

        ---
        **Example 1 (Technology):**
        "In which city is the headquarters of the company that developed the mobile operating system for the Google Pixel phone?"

        JSON desired output format:
        [
            {{
                "subquestion": "Which company develops the mobile operating system used on the Google Pixel phone?",
                "search_keyword": "company develops mobile OS Google Pixel phone"
            }},
            {{
                "subquestion": "In which city is that company's headquarters located?",
                "search_keyword": "headquarters location of Google"
            }}
        ]

        ---
        **Example 2 (Entertainment):**
        "Who directed the movie that won the Oscar for Best Original Song for the track 'My Heart Will Go On'?"

        JSON desired output format:
        [
            {{
                "subquestion": "Which movie features the Oscar-winning song 'My Heart Will Go On'?",
                "search_keyword": "movie with song My Heart Will Go On Oscar"
            }},
            {{
                "subquestion": "Who was the director of that movie?",
                "search_keyword": "director of Titanic movie"
            }}
        ]

        ---
        **Example 3 (Sports):**
        "What is the home stadium of the NFL team that quarterback Tom Brady won his last Super Bowl with?"

        JSON desired output format:
        [
            {{
                "subquestion": "Which NFL team did Tom Brady win his last Super Bowl with?",
                "search_keyword": "Tom Brady last Super Bowl winning team"
            }},
            {{
                "subquestion": "What is the name of the home stadium for that NFL team?",
                "search_keyword": "home stadium of Tampa Bay Buccaneers"
            }}
        ]

        ---
        **Example 4 (Science):**
        "What is the primary process that generates energy in the star that is closest to the planet Earth?"

        JSON desired output format:
        [
            {{
                "subquestion": "What is the name of the star closest to the planet Earth?",
                "search_keyword": "star closest to Earth"
            }},
            {{
                "subquestion": "What is the primary energy generation process in that star?",
                "search_keyword": "primary energy generation process in the Sun"
            }}
        ]

        ---
        **Example 5 (Business):**
        "What is the stock ticker symbol for the parent company of the social media platform Instagram?"

        JSON desired output format:
        [
            {{
                "subquestion": "What is the parent company of the social media platform Instagram?",
                "search_keyword": "parent company of Instagram"
            }},
            {{
                "subquestion": "What is the stock ticker symbol for that parent company?",
                "search_keyword": "stock ticker symbol for Meta Platforms"
            }}
        ]

        ---
        **Example 6 (Health):**
        "What is the common name for the phobia of the spider species whose venom can cause the condition known as latrodectism?"

        JSON desired output format:
        [
            {{
                "subquestion": "Which spider species' venom is known to cause the medical condition latrodectism?",
                "search_keyword": "spider venom causes latrodectism"
            }},
            {{
                "subquestion": "What is the common name for the phobia of that species of spiders?",
                "search_keyword": "common name for phobia of spiders"
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
    elif dataset == 'cwq':
        messages = [
          {
      "role": "system",
      "content": f"""You are a highly specialized AI assistant, an expert at decomposing complex questions into a sequence of simpler, interconnected natural language sub-questions. Your primary goal is to create a logical, step-by-step plan that faithfully represents the reasoning process to answer the main question.
      **Core Instructions for Decomposition:**
      0.  **Faithful Reasoning via Decomposition**: Your primary task is to externalize the reasoning process by breaking the `Original Complex Question` into a series of sub-questions. This decomposition itself forms the transparent and explainable path to the solution.
      1.  **Step-by-Step Plan**: Generate a sequence of sub-questions.
      2.  **Self-Contained and Searchable Sub-questions**: Each sub-question MUST be formulated so it is **self-contained**. This means it must include all necessary context from the original question or previous steps to be answerable by a standard search engine query *without needing to refer back to the original complex question or other sub-questions implicitly*. Sub-questions must be specific and unambiguous.
          * **Prefer Single Entities:** When a sub-question's answer is used in a later step, it should ideally yield a **single, specific entity** (e.g., one specific person's name, one particular place, a single defined concept). This ensures that natural language references to this entity in subsequent steps (e.g., 'that person', 'the identified location') are unambiguous.
          * **Explicit List Handling:** If an intermediate step *must* produce a list of items because the nature of the query requires it (e.g., "What known [songs] were written by X?"), any subsequent sub-question using that list **must explicitly state how to process it** (e.g., "Which of the [previously identified songs] are also [specific characteristic, e.g., 'on the album Y']?" or "What is the most frequently cited [song] from [the list found in the previous step] for [specific context]?").
          * **Avoid Vague List Generation for Intermediate Steps:** Do not formulate intermediate sub-questions that ask for "some examples" or an undefined set of items if that result is intended to be fed precisely into a subsequent step, unless that subsequent step clearly outlines how to operate on such a potentially open-ended or non-exhaustive list.
      3.  **Logical Dependency**:
          * The answer to one sub-question will often be used to formulate the subsequent sub-question.
          * When a sub-question depends on the result of a previous step, this relationship **must** be conveyed by using an explicit, concise natural language reference (e.g., "that team", "that person", "the identified author").
          * **Crucially, do not repeat the full descriptive clause from the previous step.** The goal is to create a logical chain, not a set of independent questions.
               * **Correct:** `Who was the quarterback of that team?`
               * **Incorrect:** `Who was the quarterback of the team that won Super Bowl VII?`
      4.  **Sub-question Properties**: For each step in your decomposition:
          * **`subquestion`**: State the sub-question in clear, natural language, adhering to the self-contained principle and clarity of result.
          * **`search_keyword`**: Provide a concise, keyword-based search query **template**. For any step that depends on a previous answer, this template **must** use a generic placeholder for the known entity (e.g., `author`, `team`, `composer`). The template must **not** include the specific, identifying information from the previous steps. It should only represent the *new* information being sought in relation to the placeholder.
               * **Correct:** `college attended by author`
               * **Incorrect:** `college attended by author of This Side of Paradise`
      5.  **Efficiency and Necessity & Defining Question Types**:
          * Decompose the question into the **minimum number of steps absolutely necessary** to logically arrive at the final answer.
          * **Multi-step Question Definition**: A multi-step question typically involves a chain of information retrieval where intermediate entities or properties must be resolved sequentially. One piece of information is needed to find the next, eventually leading to the final answer.
              * *Example of a multi-step question requiring decomposition*: "Which country's currency features the person who wrote 'Don Quixote'?" (This requires finding: 1. The author, 2. Their country).
          * **Single-step Question Definition**: A single-step question usually asks for a primary entity or concept (e.g., a "term", "name", "definition", "cause") that can be identified directly from a set of given characteristics, even if those characteristics are numerous or include other named entities. If all characteristics defining the target entity can be reasonably combined into a single, direct, and searchable query that is likely to yield the answer, a single step is preferred. Avoid intermediate steps that merely define well-known qualifying characteristics/entities unless defining them is absolutely essential to resolve ambiguity in the main entity being sought or if the characteristic itself is the primary unknown blocking the answer. The goal is to find the most direct path to the answer.
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
      "What films have featured the composer of Miss Nothing?"

      JSON desired output format:
      [
          {{
              "subquestion": "Who is the composer of the song 'Miss Nothing'?",
              "search_keyword": "composer of Miss Nothing"
          }},
          {{
              "subquestion": "What films has that composer been featured in?",
              "search_keyword": "films featuring composer"
          }}
      ]
      ---
      **Example 2:**
      "Where did the author of 'This Side of Paradise' go to college?"

      JSON desired output format:
      [
          {{
              "subquestion": "Who is the author of the book 'This Side of Paradise'?",
              "search_keyword": "author of This Side of Paradise"
          }},
          {{
              "subquestion": "Which college did that author attend?",
              "search_keyword": "college attended by author"
          }}
      ]
      ---
      **Example 3:**
      "Which character in Lord of the Rings was played by the actress who went to Breakwater School?"

      JSON desired output format:
      [
          {{
              "subquestion": "Which actress attended Breakwater School?",
              "search_keyword": "actress attended Breakwater School"
          }},
          {{
              "subquestion": "Which character did that actress play in the Lord of the Rings films?",
              "search_keyword": "Lord of the Rings character played by actress"
          }}
      ]
      ---
      **Example 4:**
      "What is the most recent time the winner of the 1996 NBA Finals won a championship?"

      JSON desired output format:
      [
          {{
              "subquestion": "Which team won the 1996 NBA Finals?",
              "search_keyword": "winner 1996 NBA Finals"
          }},
          {{
              "subquestion": "What was the most recent year that team also won a championship?",
              "search_keyword": "last championship year for team"
          }}
      ]
    ---
      **Example 5:**
      "What language is spoken in the place where there is a governmental office holder named Luis Fishman Zonzinski?"

      JSON desired output format:
      [
          {{
              "subquestion": "In which country or governmental jurisdiction does Luis Fishman Zonzinski hold an office?",
              "search_keyword": "country with office holder Luis Fishman Zonzinski"
          }},
          {{
              "subquestion": "What official languages are spoken in that country?",
              "search_keyword": "official languages in country"
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
                    "key_entity": "Move (1970 Film)"
                }},
                {{
                    "subquestion": "What country of citizenship is that film director from?",
                    "search_keyword": "country of citizenship director",
                    "key_entity": ""
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
                    "key_entity": "Blindfold (1966 Film)"
                }},
                {{
                    "subquestion": "When was that director born?",
                    "search_keyword": "date of birth",
                    "key_entity": ""
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
                    ""key_entity": "Helen F. Holt"
                }},
                {{
                    "subquestion": "What is the date of death of that person?",
                    "search_keyword": "date of death",
                    "key_entity": ""
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
                    "key_entity": "The Right Of The Strongest"
                    
                }},
                {{
                    "subquestion": "Where did that director die?",
                    "search_keyword": "place of death director",
                    "key_entity": ""
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
                    "key_entity": "Just Once A Great Lady (1957 Film)"
                }},
                {{
                    "subquestion": "Who was the spouse of that director?",
                    "search_keyword": "spouse of director",
                    "key_entity": ""
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
                    "key_entity": "My Friends Told Me About You"
                }},
                {{
                    "subquestion": "When was that producer born?",
                    "search_keyword": "date of birth",
                    "key_entity": ""
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
                    Here are 3 examples demonstrating the desired JSON output format and decomposition logic.
                    ---
                    **Example 1:**
                    "Which award did the director of the movie Parasite receive at the Oscars?"
                    JSON desired output format:
                    [
                    {{
                        "subquestion": "Who is the director of the movie Parasite?",
                        "search_keyword": “director of the movie 'Parasite'",
                        "key_entity": “Parasite"
                    }},
                    {{
                        "subquestion": "Which award did that director receive at the Oscars?",
                        "search_keyword": "award receive at Oscars",
                        "key_entity": ""
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
                        "key_entity": "Ada Lovelace (1815–1852)"
                    }},
                    {{
                        "subquestion": "When is that person date of birth?",
                        "search_keyword": "date of birth",
                        "key_entity": ""
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
                        "key_entity": "Doctor Strange (2016 film)"
                    }},
                    {{
                        "subquestion": "Where was director born?",
                        "search_keyword": "place of birth",
                        "key_entity": ""
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
                        "key_entity": "Web Of Suspicion"
                    }},
                    {{
                        "subquestion": "Which country of citizenship is that director from?",
                        "search_keyword": "country of citizenship",
                        "key_entity": ""
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

            The final output MUST be a JSON object. This object must contain keys: "key_entity", "key_relation".
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
                    "key_entity": "Giselda Volodi",
                    "key_relation": "sibling-in-law"
                }}
            ]

            ---
            **Example 2:**

            Question: "Who is Humphrey Of Montfort's maternal grandmother?"

            JSON desired output format:
            [
                {{
                    "key_entity": "Humphrey Of Montfort",
                    "key_relation": "maternal grandmother"
                }}
            ]

            ---
            **Example 3:**

            Question: "Who is the aunt of Margaretta Taylor?"

            JSON desired output format:
            [
                {{
                    "key_entity": "Margaretta Taylor",
                    "key_relation": "aunt"
                }}
            ]

            ---
            **Example 4:**

            Question: "What is the field of work of Fred Diamond?"

            JSON desired output format:
            [
                {{
                    "key_entity": "Fred Diamond",
                    "key_relation": "field of work"
                }}
            ]

            ---

            Now, process the following:

            Question: "{question}"
            
            Return a JSON array containing a single object. This object must contain a single key "search_keyword", whose value is the Google search query string. For example:

            [
                {{
                    "key_entity": "...",
                    "key_relation": "..."
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

def prompt_gen_answer_bioasq_step_3b(question, key_entities, context):
    prompt = f"""Question: {question}
        Instructions:
        1. Determine the question type from the question text.
        2. Generate two parts in JSON format:
        - long_answer: a detailed explanation based on the information.
        - short_answer: strictly follow these rules in order based on the question type:
        * For questions asking "how many" or about counts, return only a numeric phrase (e.g. "ten").
        * For questions asking for a range or an estimated interval, return the numeric range (e.g., "300-330 msec", "10-20%").
        * For questions asking about time or duration, return a concise time value (e.g., "36 hours", "3 days").
        * For yes/no questions, return "Yes" or "No".
        * For questions starting with "Who", "Which", or "What" that ask for a specific entity (e.g., person, disease, drug, gene, chromosome), return the full noun phrase naming that entity.
            - Examples: "chromosome X", "interleukin-6", "dopamine receptor D2".
        * For questions asking about organism, return organism type (e.g., "bacterium", "fungus").
        * If none of the above rules apply, provide a concise list of comma-separated items or a brief phrase that directly answers the question.


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


def prompt_gen_answer_cwq_step_3b(question, key_entities, context):
    prompt = f"""Question: {question}
     Instructions:
     1. Analyze the question and the provided information to determine the answer.
     2. Generate a response in a single, valid JSON object with two keys: "long_answer" and "short_answer".
     - long_answer: A detailed explanation synthesized from the information.
     - short_answer: A single string with the direct answer. Strictly follow these rules:
         * **1. Yes/No Questions:** For questions requiring a "yes" or "no" answer, the answer must be "Yes" or "No".
         * **2. Deconstruct Compound Answers:** If an answer consists of multiple concepts or is a complex descriptive phrase (like a form of government, a type of art, ...), you MUST deconstruct it into its fundamental components. Join these individual components into a single string, separated by a pipe character "|".
             - *Example 1:* For “Presidential unitary democratic republic”, the answer should be ["Presidential|Unitary|Democratic"].
             - *Example 2:* For “Drawing, Painting and Photography”, the answer should be ["Drawing|Painting|Photography"].
         * **3. Default - Single Answer:** For all other questions, provide only the single, most direct answer.
             - The entity type must precisely match what and how specific the question asks for.
             - *Example (Single Answer):* For "What is the capital of France?", the answer must be "Paris".
     3. Your entire response must be ONLY the JSON object. Do not add any commentary before or after it.

     Information:
     Key entities: {key_entities}
     Context: {context}

     Response (strict JSON):
     {{
     "long_answer": "detailed explanation",
     "short_answer": "answer as per rules"
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
           * **2. Default - Single Answer:** For all other questions, provide only the single, most direct answer from the information given.
               - The entity type must precisely match what and how specific the question asks for.
               - *Example (Location):* If multiple place names are given, select ONLY the city or state-level location (e.g. London from Kensington, London, UK). If no city or state is available, use ONLY the smallest location level mentioned.               - Do not add any additional information in parentheses.
           * **3. Insufficient information:** If the question cannot be answered with the given information, output exactly “Insufficient information”.

       3. Your entire response must be ONLY the JSON object. Do not add any commentary before or after it.

       Information:
       Key entities: {key_entities}
       Context: {context}

       Response (strict JSON):
       {{
       "long_answer": "detailed explanation",
       "short_answer": "answer as per rules"
       }}
       """
    elif type_q == "inference":
        prompt = f"""Question: {question}
        Instructions:
        1. Analyze the provided Information (Key entities, Context) to answer the Question.
        2. Your entire response MUST be a single, valid JSON object and nothing else. Do not add any commentary before or after it.
        3. The JSON object must contain two keys: "long_answer" and "short_answer".
           - "long_answer": A detailed explanation synthesized from the information.
           - "short_answer": A single string with the direct answer, generated by strictly following the Rules for Short Answer below in order.
        ---
        RULES FOR SHORT ANSWER:
        ---
        Apply these rules in the following order of priority:

        **Rule 1: Royal & Noble Figures**
        - If the question asks for a royal or noble person, construct their name precisely as follows:
            * **A. START WITH:** The person's full name and regnal number (e.g., "Christian V", "Alexander III").
            * **B. ALWAYS INCLUDE (if available):**
                -   Territorial domain (e.g., "of Denmark", "of Russia").
                -   Numbered hereditary titles, prefixed with a comma (e.g., ", 2nd Earl of Stafford", ", 3rd Duke of Suffolk").
                -   Other key titles that are part of the common name, prefixed with a comma (e.g., ", Duke of Cieszyn").
            * **C. ALWAYS EXCLUDE:**
                -   General honorifics like "King", "Queen", "Tsar", "Sultan", "Prince", "Paduka Sri", "Countess".
            * **D. PARENTHESES RULE:**
                -   The final answer string must NOT contain parentheses.
                -   However, you MUST analyze text inside parentheses in the Context to find the correct name or title.
            * **E. Examples:**
                -  Input: "Christian V of Denmark-Norway" -> Output: "Christian V of Denmark"
                -  Input: "Queen Elizabeth II of the United Kingdom" -> Output: "Elizabeth II of the United Kingdom"
                -  Input: "Emperor Meiji of Japan" -> Output: "Meiji of Japan"
                -  Input: "Frederick William, Elector of Brandenburg" -> Output: "Frederick William, Elector of Brandenburg"

        **Rule 2: All Other Questions (Default Rule)**
        - For any question not covered by the rules above, provide only the single, most direct answer found in the information.
        - The entity type must match what the question asks for.
        
        **Rule 3: Insufficient Information**
        - If the answer cannot be found in the provided information, the value must be exactly "Insufficient information".
        ---
        Information:
        Key entities: {key_entities}
        Context: {context}

        Response (strict JSON only):
        {{
            "long_answer": "detailed explanation",
            "short_answer": "answer as per rules"
        }}
        """
    elif type_q == "compositional":
        prompt = f"""Question: {question}
        Instructions:
        Your task is to act as an expert intelligence analyst. Your primary goal is to provide the **single most probable answer** to the question. Generate a single, valid JSON object.

        ### MASTER INSTRUCTION: SYNTHESIZE ALL CONTEXT ###
        This is the most important rule. The `context` you receive is a LIST of multiple documents. You are REQUIRED to:
        1.  **Read all provided documents** to understand the full picture.
        2.  **Actively connect clues between documents.** The final answer is often found by combining a piece of information from one document with a fact from another.
        3.  **Treat Synonyms as the Same Entity:** If Document A mentions "John Doe" and Document B states "Jean Doe is often credited as John Doe", you MUST recognize they are the same person.
            * **Crucial Example:** To find the mother of "John Daumery" (director of "Blind Spot"), you must connect the "Blind Spot" document (which names him) with the "Jean Daumery" document (which links the names and names the mother, "Carrie Daumery").

        ### CORE DIRECTIVE ###
        You must determine if the answer is **Explicit** (directly stated) or **Inferred** (logically deduced). An inferred answer from strong evidence is REQUIRED and is considered a correct answer. Do not be overly cautious.

        ### ANSWER DERIVATION RULES ###
        1.  **Search for an Explicit Answer First:** If the context directly states the answer, use that information.

        ### ANSWER DERIVATION RULES ###
        1.  **Search for an Explicit Answer First:** If the context directly states the answer, use that information.

        2.  **NEW: Tie-Breaker for Multiple Explicit Answers:** If you find multiple valid, explicit answers for the same question (e.g., a performer has won multiple awards), you MUST apply a hierarchy to select the single best answer.
            * **For Awards:** An induction into a **Hall of Fame** or a **Lifetime Achievement Award** MUST be prioritized over any other annual or one-time award. If awards are of similar stature, select the most recent one.

        3.  **MANDATORY INFERENCE (If no explicit answer):** If no explicit answer exists, you MUST provide an inferred answer if there are strong clues. Strong clues include, in order of priority:
            * **(i) Long-Term Residence/Work:** This is the strongest clue. If a person lived or worked in a location for most of their life, that is the inferred answer.
            * **(ii) Implied Professional Roles:** A related creative or leadership role is a strong clue if the specific role in the question is not mentioned.
                * *Example:* For an old film, if the context only mentions "screenplay by Alfred Sterling" and no director is named, "Alfred Sterling" IS the inferred director.
            * **(iii) Place of Birth (as Fallback):** If the only location clue is a birthplace and there's no mention of migration, that is the inferred answer.

        4.  **Insufficient information:** If the question cannot be answered with the given information, output exactly “Insufficient information”.

        ### OUTPUT FORMATTING (CRITICAL) ###
        After finding the answer, the `short_answer` MUST be formatted to be as concise and specific as possible by following these non-negotiable principles:

        1.  **For Locations, Extract the MOST Specific Entity:**
            * This is the highest priority formatting rule. You MUST return ONLY the smallest unit mentioned (the specific place, town, or city). Discard all larger regions like states or countries.
            * ***Examples:*** **"Valoria, Republic of Eldoria" -> "Valoria"; "Silver Creek, Lunara Province" -> "Silver Creek"; "Starfall Ranch, Starlight City" -> "Starfall Ranch".**

        2.  **For General Terms, Use the Broadest Category:**
            * ***Example:*** Use "cancer" instead of "bowel cancer".
        
        3.  **For Nationalities, follow this strict priority order:**
            * **(i) For patterns like "X-born Y" (e.g., "Spanish-born Brazilian"), the answer is ALWAYS "X".**
            * **(ii) For hyphenated nationalities like "X-Y" (e.g., "Mexican-American"), the answer is "X".**
            * **(iii) For non-hyphenated dual nationalities like "X Y" (e.g., "German Italian"), the answer is "Y".**

        ---
        Information:
        Key entities: {key_entities}
        Context: {context}

        Response (strict JSON):
        {{
        "long_answer": "Explain your reasoning, stating clearly if the answer was explicit or inferred from evidence, and which documents were used.",
        "short_answer": "The single most probable answer, formatted according to the critical output rules."
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


def prompt_gen_evidences(context):
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
    
    relations_str = ", ".join(sorted(list(allowed_relations)))

    system_prompt = f"""You are an AI expert in structured data extraction. Your task is to analyze the user's input, parse each line, and convert it into a single, valid JSON object of factual triples.

        **INPUT FORMAT:**
        `[index] entity_1; question; entity_2`

        **OUTPUT FORMAT:**
        A single JSON object: `{{"evidences": [[Entity1, Relation, Entity2], ...]}}`

        **TRANSFORMATION RULES:**
        1. Extract `Entity1` and `Entity2`.
        2. Extract `Relation` from the `question`. The `Relation` MUST BE one of the following:
           `{relations_str}`
        3. **CRITICAL RULE FOR COUNTRIES:** Your choice of relation for country-related questions depends on the **type of the entity**:
           - For **Creative Works** (films, books): A question about their country implies their production origin -> Use **`country of origin`**.
           - For **People** (directors, senators, etc.): A question about their country or nationality implies citizenship -> Use **`country of citizenship`**.
           - For **Geographical Locations** (mountains, villages, etc.): A question with "located in" implies physical placement -> Use **`country`**.
        4. If no relation fits, use `null`.
        """

    user_prompt = f"""Here are some examples. Based on these, process the final `INPUT CONTEXT`.
        **EXAMPLE 1 (Basic):**
        * INPUT CONTEXT:
            ```
            [0] Mina Gerhardsen; Who is the father of Mina Gerhardsen?; Rune Gerhardsen
            ```
        * OUTPUT JSON:
            ```json
            {{
                "evidences": [["Mina Gerhardsen", "father", "Rune Gerhardsen"]]
            }}
            ```
            
        **EXAMPLE 2 (Rule for Locations -> `country`):**
        * INPUT CONTEXT:
            ```
            [0] Puka Q'Asa (Huancavelica); In which country is this mountain located?; Peru
            ```
        * OUTPUT JSON:
            ```json
            {{
                "evidences": [["Puka Q'Asa (Huancavelica)", "country", "Peru"]]
            }}
            ```

        **EXAMPLE 3 (Rule for Creative Works -> `country of origin`):**
        * INPUT CONTEXT:
            ```
            [0] Shadows of the West; What is the country of this movie?; United States
            ```
        * OUTPUT JSON:
            ```json
            {{
                "evidences": [["Shadows of the West", "country of origin", "United States"]]
            }}
            ```

        **EXAMPLE 4 (Rule for People -> `country of citizenship`):**
        * INPUT CONTEXT:
            ```
            [0] Fernand Backes; What is this person's nationality?; Luxembourgian
            [1] Tom Carew; What country is this person from?; Irish
            ```
        * OUTPUT JSON:
            ```json
            {{
                "evidences": [
                    ["Fernand Backes", "country of citizenship", "Luxembourgian"],
                    ["Tom Carew", "country of citizenship", "Irish"]
                ]
            }}
            ```
            
        ---
        **YOUR TASK:**

        Now, process the following `INPUT CONTEXT` and generate the corresponding `OUTPUT JSON`.

        * INPUT CONTEXT:
            ```
            {context}
            ```
        * OUTPUT JSON:
        """

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]
    
    return messages


def prompt_split_question(question, type_q=None):
    system_prompt = """You are an AI assistant that splits a complex comparative question into a list of two simple sub-questions. Your response must only be a valid JSON array of two strings."""

    if type_q == "bridge_comparison":
        user_prompt = f"""
        ### EXAMPLE 1: Complex question: "Do both directors of films The Bird of Happiness (film) and Bunco Squad share the same nationality?"

        JSON array result:
        [
        "What is the nationality of the director of The Bird of Happiness (film)?",
        "What is the nationality of the director of Bunco Squad?"
        ]

        ### EXAMPLE 2: Complex question: "Which film has the director who is older than the other, Tiny Times 3 or Being Rose?"

        JSON array result:
        [
        "When was the director of Tiny Times 3 born?",
        "When was the director of Being Rose born?"
        ]
        
        ### TASK: Complex question: "{question}"
        
        JSON array result:
        """
    else:
        user_prompt = f"""
        ### EXAMPLE 1: Complex question: "Are both high schools, York High School (Virginia) and Cornwall Collegiate and Vocational School, located in the same country?"
        
        JSON array result:
        [
            "What is the location of York High School (Virginia)?",
            "What is the location of Cornwall Collegiate and Vocational School?"
        ]
        
        ### EXAMPLE 2: Complex question: "Who has a wider scope of profession, Axel Siefer or John Judge (Politician)?"
        
        JSON array result:
        [
            "What is the profession of Axel Siefer?",
            "What is the profession of John Judge (Politician)?"    
        ]
    
        ### TASK: Complex question: "{question}"
        
        JSON array result:
        """
        
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

    return messages