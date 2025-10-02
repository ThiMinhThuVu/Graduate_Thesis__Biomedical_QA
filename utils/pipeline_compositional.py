import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, Tuple

import requests
import tqdm


def setup_logging(log_file: str = 'scen1/pipeline.jsonl') -> logging.Logger:
    """Set up JSON Lines logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Clear the log file at the start of each run
    with open(log_file, 'w') as f:
        pass  # This creates an empty file or clears existing content
    
    # Create custom formatter for JSON Lines
    class JSONLFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage()
            }
            # Only add extra fields that are not standard LogRecord attributes
            reserved_attrs = {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'getMessage',
                'exc_info', 'exc_text', 'stack_info', 'taskName'
            }
            for key, value in record.__dict__.items():
                if key not in reserved_attrs:
                    log_entry[key] = value
            return json.dumps(log_entry)
    
    # Set up logger
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for JSON Lines
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(JSONLFormatter())
    logger.addHandler(file_handler)
    
    # Console handler for human-readable output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger


# Initialize logger
logger = setup_logging()


def search_wikidata_entity(entity_name: str, limit=10, max_retries=3, base_delay=1) -> list:
    """Search Wikidata entities using the Wikidata API."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json",
        "limit": limit
    }
    headers = {"User-Agent": "WikidataClient/1.0"}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [item["id"] for item in data.get("search", [])]
            
        except requests.exceptions.RequestException as e:
            logger.warning("Request error", extra={
                'attempt': attempt + 1,
                'max_retries': max_retries,
                'entity_name': entity_name,
                'error': str(e)
            })
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info("Retrying search request", extra={
                    'delay_seconds': delay,
                    'entity_name': entity_name
                })
                time.sleep(delay)
            else:
                logger.error("Max retries reached for entity search", extra={
                    'entity_name': entity_name
                })
                return []
    
    return []


def get_wikidata_entity(qid: str) -> Optional[dict]:
    """Fetch data for a Wikidata QID."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "format": "json",
    }
    headers = {"User-Agent": "WikidataClient/1.0"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching Wikidata entity", extra={
            'qid': qid,
            'error': str(e)
        })
        return None
    

def load_cache(cache_file: str = 'scen1/wikidata_cache.json') -> dict:
    """Load cached Wikidata results from JSON file."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return {}


def save_cache(cache: dict, cache_file: str = 'scen1/wikidata_cache.json') -> None:
    """Save cache to JSON file."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)


def get_attributes_by_id(qid: str, cache: Optional[dict] = None) -> dict:
    """
    Fetch attributes for a Wikidata item using SPARQL query.
    Returns dict with properties as keys and lists of values.
    Uses cache if provided to avoid redundant API calls.
    """
    # Check cache first
    if cache and qid in cache:
        logger.info("Cache hit for Wikidata entity", extra={'qid': qid})
        return cache[qid]
    
    endpoint_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?propertyLabel ?valueLabel WHERE {{
      wd:{qid} ?p ?value.
      ?property wikibase:directClaim ?p.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "WikidataClient/1.0"
    }

    try:
        response = requests.get(endpoint_url, params={"query": query}, headers=headers)
        response.raise_for_status()
        data = response.json()
        bindings = data.get("results", {}).get("bindings", [])

        if not bindings:
            logger.warning("No SPARQL results found", extra={'qid': qid})
            result = {}
        else:
            # Group multiple values for the same property
            results_dict = defaultdict(list)
            for item in bindings:
                prop = item.get("propertyLabel", {}).get("value")
                val = item.get("valueLabel", {}).get("value")
                
                if prop and val and ' ID' not in prop and val not in results_dict[prop]:
                    results_dict[prop].append(val)
            
            logger.info("Retrieved entity attributes", extra={
                'qid': qid,
                'num_properties': len(results_dict)
            })
            
            result = dict(results_dict)

        # Update cache if provided
        if cache is not None:
            cache[qid] = result
            
        return result

    except requests.exceptions.RequestException as e:
        logger.error("SPARQL request failed", extra={
            'qid': qid,
            'error': str(e)
        })
        return {}
    

def call_gemini_api(system_prompt: str, user_prompt: str, model: str, api_key: str, temperature: float = 0.7) -> Optional[str]:
    """
    Call Google Gemini API with specified parameters.
    
    Returns generated text or None if error occurs.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [{
            "parts": [{"text": user_prompt}]
        }],
        "generationConfig": {"temperature": temperature}
    }

    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        # Extract generated text from nested response structure
        candidate = result.get('candidates', [{}])[0]
        text = candidate.get('content', {}).get('parts', [{}])[0].get('text')
        
        if text:
            return text
        else:
            logger.error("Could not find generated text in API response", extra={
                'model': model,
                'response_structure': list(result.keys()) if result else None
            })
            return None

    except requests.exceptions.RequestException as e:
        logger.error("Gemini API call failed", extra={
            'model': model,
            'error': str(e)
        })
        return None
    except (KeyError, IndexError) as e:
        logger.error("Error parsing Gemini API response", extra={
            'model': model,
            'error': str(e)
        })
        return None
    

def build_prompt(entity_data: dict, step_data: Any, search_keyword: str) -> Tuple[str, str]:
    """Build prompt for Gemini API using entity data and search keyword."""
    system_prompt = "You are an expert at comparing information from 2 different sources."

    user_prompt = f"""Given the following 2 information sources about an entity:

    Model Output:
    {json.dumps(step_data, indent=2)}

    Wikidata Information:
    {json.dumps(entity_data, indent=2)}

    Decide if these 2 sources are consistent and related to each other.
    If they are, output exactly a single word:
    CONSISTENT
    If they are not, output exactly a single word:
    MISMATCHED
    """
    
    return system_prompt, user_prompt


def build_fallback_prompt(context: dict) -> Tuple[str, str]:
    """Build fallback prompt to answer the question with given context."""
    system_prompt = "You are an expert at providing concise question answering tasks."

    user_prompt = f"""Given the following context:

    Question: {context.get('question')}
    Key Entity: {context.get('key_entity')}
    Search Keyword: {context.get('search_keyword')}
    Wikidata QID: {context.get('qid')}
    Wikidata Entity Data: {json.dumps(context.get('entity_data'), indent=2)}

    Provide a concise answer to the question based on the context above without explanation.
    Answer:
    """
    
    return system_prompt, user_prompt


def test():
    """Test function for Wikidata entity search and API calls."""
    key_entity = 'Ken Burns'
    search_keyword = 'place of birth'
    api_key = 'YOUR_API_KEY_HERE'  # Replace with actual API key
    
    # Load cache
    cache = load_cache()
    
    results = search_wikidata_entity(key_entity)
    logger.info("Wikidata search completed", extra={
        'key_entity': key_entity,
        'num_results': len(results),
        'results': results
    })
    
    entity_data = {}
    if results:
        entity_data = get_attributes_by_id(results[0], cache)
        # Save cache after fetching
        save_cache(cache)
        
        with open('scen1/entity_data.json', 'w') as f:
            json.dump(entity_data, f, indent=2)

    try:
        with open('scen1/step.json', 'r') as f:
            step_data = json.load(f)
    except FileNotFoundError:
        logger.error("Test data file not found", extra={
            'file': 'scen1/step.json'
        })
        return
    
    # Only proceed if we have entity data
    if entity_data:
        model = 'gemini-2.5-flash-lite'
        system_prompt, user_prompt = build_prompt(entity_data, step_data, search_keyword)
        response = call_gemini_api(system_prompt, user_prompt, model, api_key, temperature=0)
        logger.info("Gemini API test completed", extra={
            'key_entity': key_entity,
            'response': response
        })
    else:
        logger.warning("No entity data available for API call test")


def main(k: int = 3, n_samples: int = 3000, override: bool = False):
    """
    Main function to process compositional results with Wikidata verification.
    
    Args:
        k: Number of top entities to check before stopping (default: 3)
        n_samples: Number of files to process (default: 3000)
        override: If True, reprocess files even if already marked as consistent (default: False)
    """
    api_key = 'AIzaSyDHx-fksiscyBJfa-YXrfgCd-42FJZWqUk'  # Replace with actual API key
    model = 'gemini-2.5-flash-lite'
    result_dir = '/Users/trandinhquangminh/Codespace/2wiki/compositional-2618q/wiki_compositional_results/all_ans'
    
    # Load cache at startup
    cache = load_cache()
    
    if override:
        logger.info("Running in override mode - will reprocess all files", extra={
            'override': override
        })
    
    if not os.path.exists(result_dir):
        logger.error("Result directory not found", extra={
            'result_dir': result_dir
        })
        return
    
    result_files = [f for f in os.listdir(result_dir) if f.endswith('.json')]
    result_files.sort()  # Ensure consistent processing order

    if n_samples > len(result_files):
        logger.warning("n_samples exceeds available files, adjusting to available count", extra={
            'requested_n_samples': n_samples,
            'available_files': len(result_files)
        })
        n_samples = len(result_files)

    for filename in tqdm.tqdm(result_files[2369:n_samples], desc="Processing files", disable=False):
        # Use tqdm.write() for logging to avoid interference with progress bar
        # tqdm.write(f"Processing: {filename}")
        logger.info("Processing file", extra={'file': filename})
        filepath = os.path.join(result_dir, filename)
        
        try:
            with open(filepath, 'r') as infile:
                result_data = json.load(infile)
                if not override and 'consistent' in result_data and result_data['consistent']:
                    logger.info("Skipping already processed file", extra={
                        'file': filename,
                        'reason': 'already_consistent',
                        'override': override
                    })
                    continue
                elif override and 'consistent' in result_data and result_data['consistent']:
                    logger.info("Overriding already processed file", extra={
                        'file': filename,
                        'reason': 'override_requested',
                        'override': override
                    })
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error("Error reading file", extra={
                'file': filename,
                'error': str(e)
            })
            continue
        
        if '1' not in result_data or 'final_qi' not in result_data['1']:
            logger.warning("Invalid file structure", extra={
                'file': filename,
                'missing_keys': ['1', 'final_qi']
            })
            continue
            
        model_output = result_data['1']['final_qi']
        key_entity = model_output.get('key_entity')
        search_keyword = model_output.get('search_keyword')
        
        if not key_entity or not search_keyword:
            logger.warning("Missing required fields", extra={
                'file': filename,
                'key_entity': key_entity,
                'search_keyword': search_keyword
            })
            continue
        
        wikidata_entities = search_wikidata_entity(key_entity)
        if not wikidata_entities:
            logger.warning("No Wikidata entities found", extra={
                'key_entity': key_entity,
                'file': filename
            })
            continue
        
        logger.info("Starting entity verification", extra={
            'key_entity': key_entity,
            'total_entities': len(wikidata_entities),
            'checking_count': min(k, len(wikidata_entities)),
            'file': filename
        })
        
        # Check first k entities with early stopping
        consistent_found = False
        for i, qid in enumerate(wikidata_entities[:k]):
            logger.info("Checking entity", extra={
                'entity_index': i + 1,
                'total_checking': min(k, len(wikidata_entities)),
                'qid': qid,
                'key_entity': key_entity
            })
            
            entity_data = get_attributes_by_id(qid, cache)
            if not entity_data:
                logger.warning("No entity data found", extra={
                    'qid': qid,
                    'key_entity': key_entity
                })
                continue
                
            system_prompt, user_prompt = build_prompt(entity_data, model_output, search_keyword)
            response = call_gemini_api(system_prompt, user_prompt, model, api_key, temperature=0)
            
            logger.info("Entity verification result", extra={
                'key_entity': key_entity,
                'qid': qid,
                'response': response,
                'file': filename
            })
            
            if response and response.strip().upper() == 'CONSISTENT':
                logger.info("Consistent match found", extra={
                    'key_entity': key_entity,
                    'qid': qid,
                    'file': filename
                })
                result_data['consistent'] = True
                result_data['normalized_at_k'] = i + 1
                # Save normalized wikidata info
                if search_keyword in entity_data and entity_data[search_keyword]:
                    result_data['normalized_wikidata'] = entity_data[search_keyword][0]  # Only take the first value
                else:
                    result_data['normalized_wikidata'] = None
                    logger.info("No exact match for search keyword, using fallback", extra={
                        'search_keyword': search_keyword,
                        'available_properties': list(entity_data.keys()) if entity_data else [],
                        'key_entity': key_entity
                    })
                    fallback_context = {
                        'question': model_output['question'],
                        'key_entity': key_entity,
                        'search_keyword': search_keyword,
                        'qid': qid,
                        'entity_data': entity_data
                    }
                    fallback_system_prompt, fallback_user_prompt = build_fallback_prompt(fallback_context)
                    fallback_response = call_gemini_api(fallback_system_prompt, fallback_user_prompt, model, api_key, temperature=0)
                    if fallback_response:
                        result_data['genai_fallback'] = fallback_response
                        logger.info("Fallback response generated", extra={
                            'key_entity': key_entity,
                            'fallback_response_length': len(fallback_response)
                        })
                    else:
                        logger.warning("Fallback response failed", extra={
                            'key_entity': key_entity
                        })
                result_data['entity_data'] = entity_data
                consistent_found = True
                with open(filepath, 'w') as outfile:
                    json.dump(result_data, outfile, indent=4)
                break  # Early stopping
            else:
                logger.info("No match for entity", extra={
                    'qid': qid,
                    'key_entity': key_entity
                })
        
        if not consistent_found:
            logger.warning("No consistent match found", extra={
                'key_entity': key_entity,
                'entities_checked': k,
                'file': filename
            })
    
    # Save cache after processing all files
    save_cache(cache)
    logger.info("Processing completed", extra={
        'cache_entries': len(cache),
        'files_processed': min(n_samples, len(result_files))
    })


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    k = 3  # default value
    override = False  # default value
    
    # Check for override flag
    if '--override' in sys.argv:
        override = True
        sys.argv.remove('--override')  # Remove it so it doesn't interfere with k parsing
        logger.info("Override mode enabled", extra={
            'override': override
        })
    
    # Parse k value
    if len(sys.argv) > 1:
        try:
            k = int(sys.argv[1])
            logger.info("Configuration set", extra={
                'k_entities': k,
                'override': override,
                'source': 'command_line'
            })
        except ValueError:
            logger.warning("Invalid k value provided, using default", extra={
                'provided_value': sys.argv[1],
                'default_k': 3,
                'override': override
            })
    
    main(k, override=override)