import requests
import time
from src.api_key import API_LIST_4O, API_LIST_O3, API_KEY_GG, API_KEY_serp

API_URL = "https://cf.mnapi.com/v1/chat/completions"  

current_api_index = 0

def request_api(messages, model, temperature=0):
    global current_api_index
    max_retries = 50
    retry_delay = 30
    
    if model == "gpt-4o-mini":
        API_LIST = API_LIST_4O
        max_tokens = 4096
    elif model == "o3-mini":
        API_LIST = API_LIST_O3
        max_tokens = 8192
        
    data = {
        "model": model, 
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(max_retries):
        api = API_LIST[current_api_index]

        headers = {
            "Authorization": f"Bearer {api}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            response = requests.post(API_URL, headers=headers, json=data)
            
            if response.status_code == 200:
                response = response.json()

                if "choices" in response and len(response["choices"]) > 0:
                    result_json = response["choices"][0].get("message", {}).get("content", "")
                    if result_json == "":
                        print(f"Empty response from API {api}. Retrying...")
                    return result_json
                
                else:
                    raise ValueError("Unexpected response format from API: 'choices' not found or empty.")
            else:
                print(f"API {api} failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error with API {current_api_index}: {e}")

        current_api_index = (current_api_index + 1) % len(API_LIST)
        print(f"Retrying with next API in {retry_delay} seconds...")
        time.sleep(retry_delay)

    raise RuntimeError("All APIs failed after multiple retries.")

def search_google(query):
    max_retries = 10
    retry_delay = 2
   
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'https://api.search1api.com/search',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': API_KEY_GG 
                },
                json={
                    "query": query,
                    "search_service": "google",
                    "max_results": 10,
                    "crawl_results": 0,
                    "image": False,
                    "language": "en"
                },
                timeout=30
            )

            if response.status_code in [200, 404]:
                return response.json().get("results", [])
            else:
                raise RuntimeError(f"Search API failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e  

from serpapi import GoogleSearch

def serapi_search_google(
    query: str,
    num_results: int = 10,
    language: str = "en",
    country: str = "us"
):
    """
    Search Google using SerpAPI.

    Args:
        query (str): Search query
        api_key (str): SerpAPI API key
        num_results (int): Number of results to return
        language (str): Language code (default: en)
        country (str): Country code (default: us)

    Returns:
        list[dict]: Search results (title, snippet, link)
    """

    params = {
        "engine": "google",
        "q": query,
        "api_key": API_KEY_serp,
        "hl": language,
        "gl": country,
        "num": num_results
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    output = []
    for item in results.get("organic_results", []):
        output.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link")
        })

    return output

if __name__ == "__main__":    
    # result = search_google("Is cabergoline safe to use for a patient with cholestasis who has a prolactinoma?")
    question = "Where did that person graduate from?"
    key_entities = "James Scullin"
    context = """
James Scullin: James Henry Scullin (18 September 1876 – 28 January 1953) was an Australian Labor Party politician and the ninth Prime Minister of Australia. Scullin led Labor to government at the 1929 election. The Wall Street Crash of 1929 transpired just two days after his swearing in, which would herald the beginning of the Great Depression in Australia. Scullin's administration would soon be overwhelmed by the economic crisis, with interpersonal and policy disagreements causing a three-way split of his party that would bring down the government in late 1931. Despite his chaotic term of office, Scullin remained a leading figure in the Labor movement throughout his lifetime, and served as an "éminence grise" in various capacities for the party until his retirement in 1949. The son of working-class Irish-immigrants, Scullin spent much of his early life as a laborer and grocer in Ballarat. An autodidact and passionate debater, Scullin would join the Australian Labor Party in 1903, beginning a career spanning five decades. He was a political organizer and newspaper editor for the party, and was elected to the House of Representatives first in 1910 and then again in 1922 until 1949. Scullin quickly established himself as a leading voice in parliament, rapidly rising to become deputy leader of the party in 1927 and then Leader of the Opposition in 1928. After Scullin had won a landslide election in 1929, events took a dramatic change with the crisis on Wall Street and the rapid onset of the Great Depression around the world, which hit heavily indebted Australia hard. Scullin and his Treasurer Ted Theodore responded by developing several plans during 1930 and 1931 to repay foreign debt, provide relief to farmers and create economic stimulus to curb unemployment based on deficit spending and expansionary monetary policy. Although the Keynesian Revolution would see these ideas adopted by most Western nations by the end of the decade, in 1931 such ideas were considered radical and the plans were bitterly opposed by many who feared hyperinflation and economic ruin. The still opposition-dominated Senate, and the conservative-dominated boards of the Commonwealth Bank and Loan Council, repeatedly blocked the plans. With the prospect of bankruptcy facing the government, Scullin backed down and instead advanced the Premiers' Plan, a far more conservative measure that met the crisis with severe cutbacks in government spending. Pensioners and other core Labor constituencies were severely affected by the cuts, leading to a widespread revolt and multiple defections in parliament. After several months of infighting the government collapsed, and was resoundingly defeated by the newly formed United Australia Party at the subsequent 1931 election. Scullin would remain party leader for four more years but the party split would not be healed until after Scullin's return to the backbenches in 1935. Scullin became a respected elder voice within the party and leading authority on taxation and government finance, and would eventually play a significant role in reforming both when Labor returned to government in 1941. Although disappointed with his own term of office, he nonetheless lived long enough to see many of his government's ideas implemented by subsequent governments before his death in 1953.
    """

    prompt = f"""
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
        Question: {question}
        Context: {context}

        Response (strict JSON):
        {{
        "long_answer": "Explain reasoning, clearly state if explicit or inferred, and cite which clues/documents were used.",
        "short_answer": "The single most probable answer, formatted according to the output rules."
        }}
        """
        
    messages = [
            {"role": "system", "content": "You are an expert in general knowledge. Your task is to provide accurate answers based on the provided information."},
            {"role": "user", "content": prompt}
        ]
    
    result = request_api(
        # messages=[{"role": "user", "content": "What is the capital of France?"}],
        messages=messages,
        model="gpt-4o-mini"
    )
    print(result)
