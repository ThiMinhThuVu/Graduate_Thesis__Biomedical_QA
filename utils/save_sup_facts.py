import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TYPES = [
    'bridge_comparison',
    'comparison',
    'compositional',
    'inference'
]

def check_word_overlap(sentence, paragraph, threshold=0.7):
    """Check if sentence and paragraph have substantial word overlap."""
    sentence_words = set(word.lower() for word in sentence.split() if len(word) > 3)
    paragraph_words = set(word.lower() for word in paragraph.split() if len(word) > 3)
    
    if len(sentence_words) == 0:
        return False
        
    overlap = len(sentence_words.intersection(paragraph_words))
    return overlap / len(sentence_words) >= threshold

def extract_title_and_content(retrieval_info):
    """Extract title and content from retrieval_info string."""
    if ':' in retrieval_info:
        parts = retrieval_info.split(':', 1)
        title = parts[0].strip()
        content = parts[1].strip()
        return title, content
    return None, retrieval_info

def process_type(q_type, base_dir):
    """Process a single data type."""
    try:
        all_ans_dir = base_dir / q_type / f'wiki_{q_type}_results' / 'all_ans'
        sup_facts_dir = base_dir / q_type / f'wiki_{q_type}_results' / 'sup_facts'
        dev_file = base_dir / 'dataset' / 'dev' / f'dev_{q_type}.json'

        # Check if required directories exist
        if not all_ans_dir.exists():
            logger.error(f"Directory does not exist: {all_ans_dir}")
            return False
        
        if not dev_file.exists():
            logger.error(f"Dev file does not exist: {dev_file}")
            return False

        # Create output directory
        sup_facts_dir.mkdir(parents=True, exist_ok=True)

        # Load dev data
        try:
            with open(dev_file, 'r', encoding='utf-8') as f:
                dev_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load dev data from {dev_file}: {e}")
            return False

        # Process all answer files
        processed_files = 0
        for file_path in all_ans_dir.glob('*.json'):
            try:
                sup_facts_data = []

                with open(file_path, 'r', encoding='utf-8') as f:
                    all_ans_data = json.load(f)

                # Validate required fields
                if 'QIDX' not in all_ans_data:
                    logger.warning(f"Missing QIDX in {file_path}")
                    continue

                qidx = all_ans_data['QIDX']
                
                # Check if QIDX exists in dev data
                if qidx not in dev_data:
                    logger.warning(f"QIDX {qidx} not found in dev data for {file_path}")
                    continue

                # Get numeric keys and sort them
                numeric_keys = sorted([k for k in all_ans_data.keys() if k.isdigit()], key=int)
                
                for key in numeric_keys:
                    if '0' not in all_ans_data[key]:
                        continue
                        
                    # The structure is data[key]['0'][sub_key] where sub_key is another dict
                    zero_data = all_ans_data[key]['0']
                    if not isinstance(zero_data, dict):
                        continue
                        
                    for sub_key, hop_data in zero_data.items():
                        if not isinstance(hop_data, dict):
                            continue
                            
                        retrieval_info = hop_data.get('retrieval_information', '')
                        if not retrieval_info:
                            continue

                        # Extract title and content from retrieval_info
                        extracted_title, content = extract_title_and_content(retrieval_info)
                        
                        # Find matching context in dev data
                        dev_context_data = dev_data[qidx].get('context', [])
                        
                        # Split content into sentences for multi-sentence retrieval info
                        sentences = []
                        for sep in ['. ', '! ', '? ', '\n']:
                            if sep in content:
                                sentences = [s.strip() for s in content.split(sep) if s.strip()]
                                break
                        if not sentences:
                            sentences = [content.strip()]
                        
                        for ctx in dev_context_data:
                            if len(ctx) < 2:
                                continue
                                
                            title = ctx[0]
                            paragraphs = ctx[1]

                            # Check if this context matches our retrieval info title
                            title_match = (extracted_title and extracted_title.lower() == title.lower())
                            
                            if title_match:
                                # For title match, find all paragraph indices that contain the sentences
                                found_indices = set()
                                
                                for sentence in sentences:
                                    if len(sentence.strip()) < 5:  # Skip very short fragments
                                        continue
                                    
                                    for i, para in enumerate(paragraphs):
                                        # Check if sentence matches paragraph (handle formatting differences)
                                        sentence_clean = sentence.strip().rstrip('.')
                                        para_clean = para.strip().rstrip('.')
                                        
                                        if (sentence_clean.lower() in para_clean.lower() or 
                                            para_clean.lower() in sentence_clean.lower() or
                                            # Check for substantial word overlap
                                            check_word_overlap(sentence_clean, para_clean)):
                                            found_indices.add(i)
                                
                                # Add supporting facts for all found paragraph indices
                                for idx in sorted(found_indices):
                                    sup_facts_data.append([title, idx])
                                
                                break  # Found the matching title, no need to continue
                            
                            else:
                                # For non-title matches, check each sentence against paragraphs
                                for sentence in sentences:
                                    if len(sentence.strip()) < 5:
                                        continue
                                    
                                    for i, para in enumerate(paragraphs):
                                        if sentence.strip().lower() in para.lower():
                                            sup_facts_data.append([title, i])
                                            break

                # Save supporting facts
                output_file = sup_facts_dir / file_path.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(sup_facts_data, f, indent=4, ensure_ascii=False)
                
                processed_files += 1
                logger.info(f"Processed {file_path.name} -> {len(sup_facts_data)} supporting facts")

            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue

        logger.info(f"Completed processing {q_type}: {processed_files} files processed")
        return True

    except Exception as e:
        logger.error(f"Failed to process type {q_type}: {e}")
        return False

def main():
    """Main function to process all types."""
    # Get script directory and set base directory
    script_dir = Path(__file__).parent
    base_dir = script_dir  # Assumes script is in the base data directory
    
    logger.info("Starting supporting facts extraction...")
    
    success_count = 0
    for q_type in TYPES:
        logger.info(f"Processing type: {q_type}")
        if process_type(q_type, base_dir):
            success_count += 1
        else:
            logger.error(f"Failed to process type: {q_type}")
    
    logger.info(f"Completed: {success_count}/{len(TYPES)} types processed successfully")

if __name__ == "__main__":
    main()
