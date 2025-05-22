import os
import json
import argparse
import time
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import traceback

from vllm import LLM, SamplingParams

# Model path
MODEL_PATH = "/path/to/Qwen2.5-VL-72B-Instruct"

# Initialize logs directory
os.makedirs("logs", exist_ok=True)

def load_json_data(json_path):
    """Load JSON data"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def initialize_model(model_path=MODEL_PATH, tensor_parallel_size=8, max_model_len=32768):
    """Initialize model"""
    print(f"Initializing model: {model_path} using {tensor_parallel_size} GPUs...")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=256,
    )
    
    return llm, sampling_params

def extract_edit_prompt(edit_prompt, llm, sampling_params, max_retries=3):
    """Extract edit prompt and convert to declarative statement"""
    
    # Extract part starting with "Can you"
    can_you_part = ""
    match = re.search(r"Can you[^?]*\?", edit_prompt, re.IGNORECASE)
    if match:
        can_you_part = match.group(0)
    else:
        # If no match found, try to extract the last sentence
        sentences = re.split(r'[.!?]', edit_prompt)
        can_you_part = sentences[-1].strip() if sentences else edit_prompt
    
    # Build prompt
    prompt_template = """Convert the following question into a declarative statement that describes the difference between two images.

Question: "{can_you_part}"

Your answer should be in this format:
"The second image differs from the first image by [description of the change]."

Only provide the converted statement without any explanations or additional text.
"""

    prompt = prompt_template.format(can_you_part=can_you_part)
    
    # Generation with retry mechanism
    for retry in range(max_retries):
        try:
            outputs = llm.generate([prompt], sampling_params=sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            
            # If response begins with expected format, return it
            if response_text.startswith("The second image differs"):
                return response_text
            else:
                # Try to extract a sentence that matches the format
                match = re.search(r"The second image differs[^.]*\.", response_text)
                if match:
                    return match.group(0)
                else:
                    print(f"Retry {retry+1}/{max_retries}: Incorrect output format: '{response_text}'")
                    time.sleep(1)
        except Exception as e:
            print(f"Retry {retry+1}/{max_retries} failed: {str(e)}")
            time.sleep(2)
    
    # If all retries fail, return a default conversion
    return f"The second image differs from the first image by changes as requested in: {can_you_part}"

def check_description_match(second_edit_prompt, difference_description, llm, sampling_params, max_retries=3):
    """Check if descriptions match"""
    
    # Build prompt
    prompt_template = """
    Your task is determine if the 'Described Difference' matches the 'Second Description':

    Second Description: {second_edit_prompt}
    Described Difference: {difference_description}

    IMPORTANT GUIDELINES FOR EVALUATION:

    - "Caption 1" refers to the first image description, and "Caption 2" refers to the second image description.
    - FOCUS ON MEANING, NOT EXACT WORDING: Synonyms, paraphrasing, and alternative expressions that convey the same meaning should be considered equivalent.
    - ALLOW FOR VARIATION: The difference description may use different phrasing or perspective than the second description while still being correct.
    - ACCEPT STRUCTURAL DIFFERENCES: Different sentence structures, word order, or narrative approaches that communicate the same content are valid.
    - RECOGNIZE CONCEPTUAL MATCHES: When categories, concepts, or observations are semantically similar (even if using different terminology), treat them as matches.
    - EVALUATE BASED ON CORE CHANGES: If both descriptions identify the same fundamental changes between images, even through different expressions, consider them aligned.
    - BE GENEROUS IN INTERPRETATION: When in doubt, lean toward accepting descriptions that capture the essential elements, even if details vary.


    Answer with 1. Only "Yes" if they match. No explanation needed.
                2. Only "No" if there's no obvious change. No explanation needed.
"""

    prompt = prompt_template.format(
        second_edit_prompt=second_edit_prompt,
        difference_description=difference_description
    )
    
    # Generation with retry mechanism
    for retry in range(max_retries):
        try:
            outputs = llm.generate([prompt], sampling_params=sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            
            # Extract Yes or No
            if "yes" in response_text.lower():
                return "Yes"
            elif "no" in response_text.lower():
                return "No"
            else:
                print(f"Retry {retry+1}/{max_retries}: Incorrect output format: '{response_text}'")
                time.sleep(1)
        except Exception as e:
            print(f"Retry {retry+1}/{max_retries} failed: {str(e)}")
            time.sleep(2)
    
    # If all retries fail, return error
    return "Error: Unable to determine match"

def log_error(error_msg, item_id, error_file="./logs/process_error.txt"):
    """Log error information to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_file, 'a') as f:
        f.write(f"[{timestamp}] Error processing {item_id}\n")
        f.write(f"Error message: {error_msg}\n")
        f.write("-" * 80 + "\n")
    print(f"Error logged to {error_file}")

def save_checkpoint(results, processed_count, output_path, shard_id=None):
    """Save checkpoint"""
    suffix = f"_shard{shard_id}" if shard_id is not None else ""
    checkpoint_path = f"{os.path.splitext(output_path)[0]}{suffix}_checkpoint.json"
    
    checkpoint_data = {
        "processed_count": processed_count,
        "results": results
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=4)
    
    print(f"Checkpoint saved: {checkpoint_path} (Processed: {processed_count} items)")
    return checkpoint_path

def load_checkpoint(output_path, shard_id=None):
    """Load checkpoint if it exists"""
    suffix = f"_shard{shard_id}" if shard_id is not None else ""
    checkpoint_path = f"{os.path.splitext(output_path)[0]}{suffix}_checkpoint.json"
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data["results"], checkpoint_data["processed_count"]
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
    
    return [], 0

def process_data_shard(data, output_path, shard_id=None, total_shards=None, 
                      checkpoint_interval=10, tensor_parallel_size=8):
    """Process data shard"""
    # If shard specified, only process that shard
    if shard_id is not None and total_shards is not None:
        shard_size = len(data) // total_shards
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size if shard_id < total_shards - 1 else len(data)
        data_shard = data[start_idx:end_idx]
        print(f"Processing shard {shard_id+1}/{total_shards}, {len(data_shard)} items (total data: {len(data)} items)")
    else:
        data_shard = data
        print(f"Processing complete dataset, {len(data_shard)} items")
    
    # Load checkpoint
    results, start_idx = load_checkpoint(output_path, shard_id)
    
    if start_idx > 0:
        print(f"Resuming from checkpoint: {start_idx} items already processed")
    
    # Initialize model
    llm, sampling_params = initialize_model(tensor_parallel_size=tensor_parallel_size)
    
    # Process each item
    for idx, item in enumerate(tqdm(data_shard[start_idx:], initial=start_idx, total=len(data_shard)), start=start_idx):
        # Create new result item, preserve original data
        result_item = item.copy()
        
        try:
            # 1. Extract and convert edit prompt
            edit_prompt = item["edit_prompt"]
            second_edit_prompt = extract_edit_prompt(edit_prompt, llm, sampling_params)
            
            # Add converted edit prompt to result
            result_item["second_edit_prompt"] = second_edit_prompt
            
            # 2. Check if descriptions match
            difference_description = item["difference_description"]
            match_result = check_description_match(second_edit_prompt, difference_description, llm, sampling_params)
            
            # Add match result
            result_item["second_edit_judge"] = match_result
            
        except Exception as e:
            error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
            item_id = item.get("image_name", f"item_{idx}")
            log_error(error_msg, item_id)
            
            # Add error information
            result_item["second_edit_prompt"] = f"Error: {str(e)}"
            result_item["second_edit_judge"] = "Error"
        
        # Add to results list
        results.append(result_item)
        
        # Save checkpoint periodically
        if (idx + 1) % checkpoint_interval == 0:
            save_checkpoint(results, idx + 1, output_path, shard_id)
    
    # Save final results
    suffix = f"_shard{shard_id}" if shard_id is not None else ""
    final_output_path = f"{os.path.splitext(output_path)[0]}{suffix}.json"
    
    final_data = {
        "processed_count": len(results),
        "results": results
    }
    
    with open(final_output_path, 'w') as f:
        json.dump(final_data, f, indent=4)
    
    # Also save final checkpoint
    save_checkpoint(results, len(data_shard), output_path, shard_id)
    
    print(f"Completed processing shard {shard_id if shard_id is not None else 'all'}")
    print(f"Results saved to: {final_output_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process edit prompts and check description matches")
    parser.add_argument("--input_json", type=str, default='/fs-computility/llmit_d/shared/baitianyi/tldr/Data/visualgenome/filter_edit_resultsv2/negative_samples.json', help="Input JSON data path")
    parser.add_argument("--output_path", type=str, default='/fs-computility/llmit_d/shared/baitianyi/tldr/Data/visualgenome/filter_edit_resultsv2/second_caption_w_edit_judge.json', help="Output path")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint interval (save every N items)")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID to process (starting from 0)")
    parser.add_argument("--total_shards", type=int, default=8, help="Total number of shards")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Load data
    data = load_json_data(args.input_json)
    if "results" in data:
        data = data["results"]
    
    # Process specified shard or all data
    process_data_shard(
        data,
        args.output_path,
        args.shard_id,
        args.total_shards,
        args.checkpoint_interval,
        args.tensor_parallel_size
    )