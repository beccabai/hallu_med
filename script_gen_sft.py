import os
import json
import argparse
import time
import traceback
import re
from tqdm import tqdm
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model path
MODEL_PATH = "/path/to//Qwen/Qwen3-32B"

# Initialize logs directory
os.makedirs("logs", exist_ok=True)

def load_json_data(json_path):
    """Load JSON data"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def initialize_model(model_path=MODEL_PATH):
    """Initialize model"""
    print(f"Initializing model: {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    
    return model, tokenizer

def generate_with_model(model, tokenizer, prompt, max_tokens=256):
    """Generate text using Qwen3 model"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking mode
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate text
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=0.1,
        top_p=0.95
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # Decode generated text
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def rewrite_difference_description(diff_description, model, tokenizer, max_retries=3):
    """Rewrite the difference_description field"""
    
    # Build prompt (in English)
    prompt_template = """
I need your help to rewrite a description of differences between two images.

Current description (needs improvement): 
"{diff_description}"

Please rewrite this description to focus on the VISUAL DIFFERENCES between the two images, NOT just differences in captions. 

Important guidelines:
- Assume "Caption 1" refers to the first image, and "Caption 2" refers to the second image
- Your rewritten description should describe what appears in the second image compared to the first image
- Use formats like "The second image shows..." or "In the first image, ... while in the second image..."
- Focus on actual visual elements that are present or absent in each image
- Be specific about what was added, removed, or changed
- Keep your description concise but informative

Your response must follow this exact format:
IMPROVED_DESCRIPTION: [Your rewritten description here]

Do not include any other text, explanations, or formatting in your response.
"""

    prompt = prompt_template.format(diff_description=diff_description)
    
    # Generation with retry mechanism
    for retry in range(max_retries):
        try:
            response_text = generate_with_model(model, tokenizer, prompt)
            response_text = response_text.strip()
            
            # Extract formatted response
            pattern = r"IMPROVED_DESCRIPTION:\s*(.*)"
            match = re.search(pattern, response_text, re.DOTALL)
            
            if match:
                return match.group(1).strip()
            else:
                print(f"Retry {retry+1}/{max_retries}: Could not find correctly formatted output: '{response_text}'")
                time.sleep(1)
        except Exception as e:
            print(f"Retry {retry+1}/{max_retries} failed: {str(e)}")
            time.sleep(2)
    
    # If all retries fail, try to directly return response text, removing possible prefix
    if response_text and "IMPROVED_DESCRIPTION" in response_text:
        return response_text.split("IMPROVED_DESCRIPTION:", 1)[1].strip()
    
    # If still failing, return original description
    return diff_description

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

def process_data_shard(data, output_path, shard_id=None, total_shards=None, checkpoint_interval=10):
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
    model, tokenizer = initialize_model()
    
    # Process each item
    for idx, item in enumerate(tqdm(data_shard[start_idx:], initial=start_idx, total=len(data_shard)), start=start_idx):
        # Create new result item, preserve original data
        result_item = item.copy()
        
        try:
            # Only use difference_description for rewriting
            diff_description = item.get("difference_description", "")
            
            # Rewrite difference_description
            sft_description = rewrite_difference_description(diff_description, model, tokenizer)
            
            # Save rewritten content as "sft_description" field
            result_item["sft_description"] = sft_description
            
        except Exception as e:
            error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
            item_id = item.get("image_name", f"item_{idx}")
            log_error(error_msg, item_id)
            
            # On error, set sft_description to original difference_description
            result_item["sft_description"] = item.get("difference_description", "")
            result_item["error"] = f"Error: {str(e)}"
        
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
    parser = argparse.ArgumentParser(description="Rewrite difference_description field in JSON")
    parser.add_argument("--input_json", type=str, default='/fs-computility/llmit_d/shared/baitianyi/tldr/Data/visualgenome/filter_edit_resultsv2_batch_3/caption_w_edit_judge.json', help="Input JSON data path")
    parser.add_argument("--output_path", type=str, default='./sft_processed_batch3.json', help="Output path")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Checkpoint interval (save every N items)")
    parser.add_argument("--shard_id", type=int, default=3, help="Shard ID to process (starting from 0)")
    parser.add_argument("--total_shards", type=int, default=4, help="Total number of shards")
    parser.add_argument("--model_name", type=str, default="Qwen3-32B", help="Model name")
    
    args = parser.parse_args()
    
    # Update model path
    MODEL_PATH = f"/path/to//Qwen/{args.model_name}"
    
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
        args.checkpoint_interval
    )