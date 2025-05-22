import os
import json
import time
import traceback
from datetime import datetime
import re
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import signal
import argparse
from collections import defaultdict

from vllm import LLM, SamplingParams

# Model and file paths
MODEL_PATH = "path/to/deepseek-r1-distill-llama70b"
INPUT_PATH = "path/to/vg_caption_simple_english.json"
OUTPUT_PATH = "./datasets/vg_descriptions.json"
CHECKPOINT_INTERVAL = 10
MAX_RETRIES = 5

def get_description_prompt():
    base_prompt = '''Task: You will receive several separate captions that all refer to the same image.
Goal: Combine the information from every caption into one coherent paragraph that describes the image.
Restrictions:
	1.	Do not introduce or infer any details that are not explicitly stated in the captions.
	2.	Do not omit any information that is present in the captions.
	3.	The final paragraph must only contain elements found in the given captions and must stay strictly within their scope.
Output format: A single, well‑structured English paragraph that merges the captions into a unified description.'''
    
    return base_prompt

def initialize_model(model_path=MODEL_PATH, tensor_parallel_size=4, max_model_len=32768, batch_size=8):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        max_num_seqs=batch_size,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.92,
        top_k=40,
        min_p=0,
        repetition_penalty=1.1,
        max_tokens=4096,
    )
    
    return llm, sampling_params

def extract_description(text):
    # Remove think tokens
    if "</think>" in text:
        split_text = text.split('</think>')
        text = split_text[1].strip()
    
    # Clean any remaining think tags with regex
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    return text.strip()

def validate_description(text):
    """Validate the generated description"""
    if "<think>" in text or "</think>" in text:
        return False, "Response still contains think tags"
    
    if not text.strip():
        return False, "Empty response"
    
    if len(text.strip()) < 10:
        return False, "Response too short"
    
    return True, ""

def log_error(error_msg, image_id, idx, error_file="description_generation_error.txt"):
    """Log error information to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(error_file, 'a') as f:
            f.write(f"[{timestamp}] Error processing item {idx}: {image_id}\n")
            f.write(f"Error message: {error_msg}\n")
            f.write("-" * 80 + "\n")
        print(f"Error logged to {error_file}")
    except Exception as e:
        print(f"Failed to write to error log: {str(e)}")

def ensure_directory_exists(file_path):
    """Ensure the directory for the file path exists"""
    directory = os.path.dirname(os.path.abspath(file_path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    return directory

def save_checkpoint(results, processed_count, all_image_ids, output_path):
    """Save checkpoint data"""
    checkpoint_path = f"{os.path.splitext(output_path)[0]}_checkpoint.json"
    checkpoint_data = {
        "processed_count": processed_count,
        "results": results,
        "all_image_ids": all_image_ids,
        "last_update": time.time()
    }
    
    ensure_directory_exists(checkpoint_path)
    
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False)
        print(f"Checkpoint saved: {checkpoint_path} (Processed: {processed_count} items)")
        return checkpoint_path
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        traceback.print_exc()
        
        # Try saving to current directory as fallback
        try:
            fallback_path = f"./checkpoint_fallback_{int(time.time())}.json"
            with open(fallback_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False)
            print(f"Fallback checkpoint saved to: {fallback_path}")
            return fallback_path
        except Exception as fallback_error:
            print(f"Failed to save fallback checkpoint: {str(fallback_error)}")
            return None

def load_checkpoint(output_path):
    """Load checkpoint if it exists"""
    checkpoint_path = f"{os.path.splitext(output_path)[0]}_checkpoint.json"
    if os.path.exists(checkpoint_path):
        try:
            print(f"Found checkpoint at {checkpoint_path}, loading...")
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            print(f"Loaded checkpoint with {checkpoint_data['processed_count']} processed items")
            return checkpoint_data["results"], checkpoint_data["processed_count"], checkpoint_data.get("all_image_ids", [])
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            traceback.print_exc()
    
    print("No checkpoint found or error loading checkpoint, starting from scratch")
    return [], 0, []

def setup_signal_handlers():
    """Set up signal handlers for safe termination"""
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, safely exiting...")
        print("Program will exit after current batch completes, checkpoint saved")
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def group_captions_by_image_id(input_data):
    """Group input data by image_id"""
    grouped_data = defaultdict(list)
    for item in input_data:
        image_id = item["image_id"]
        grouped_data[image_id].append(item)
    
    print(f"Grouped data into {len(grouped_data)} unique image IDs")
    return grouped_data

def process_descriptions(input_path, output_path, checkpoint_interval=CHECKPOINT_INTERVAL,
                         tensor_parallel_size=4, batch_size=8):
    """Process data in batches and generate descriptions"""
    ensure_directory_exists(output_path)
    
    # Load checkpoint
    results, start_idx, processed_image_ids = load_checkpoint(output_path)
    processed_image_ids_set = set(processed_image_ids)
    
    if start_idx > 0:
        print(f"Resuming from checkpoint: {start_idx} items already processed.")
        print(f"Already processed {len(processed_image_ids_set)} unique image IDs.")
    
    # Initialize model
    print(f"Initializing model {MODEL_PATH} with {tensor_parallel_size} GPUs...")
    llm, sampling_params = initialize_model(
        model_path=MODEL_PATH,
        tensor_parallel_size=tensor_parallel_size,
        batch_size=batch_size
    )
    
    system_prompt = get_description_prompt()
    
    # Load input data
    print(f"Loading input data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    print(f"Loaded {len(input_data)} input items")
    
    # Group by image_id
    grouped_data = group_captions_by_image_id(input_data)
    
    # Filter out already processed image_ids
    image_ids = [img_id for img_id in grouped_data.keys() if img_id not in processed_image_ids_set]
    total_items = len(image_ids)
    
    print(f"Found {total_items} unique images to process")
    
    # Process data in batches
    with tqdm(total=total_items, desc="Processing items") as pbar:
        if start_idx > 0:
            pbar.update(start_idx)
        
        for batch_start in range(0, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            batch_image_ids = image_ids[batch_start:batch_end]
            
            # Prepare prompts for batch
            prompts = []
            batch_info = []
            
            for i, image_id in enumerate(batch_image_ids):
                try:
                    items = grouped_data[image_id]
                    
                    # Extract captions
                    captions = [item["caption"] for item in items]
                    image_path = items[0]["image"]
                    dataset = items[0]["dataset"]
                    
                    # Build caption text
                    captions_text = "\n".join([f"Caption {j+1}: {caption}" for j, caption in enumerate(captions)])
                    
                    # Create user prompt
                    user_prompt = f'''Here are several captions describing the same image:

{captions_text}

Please combine these captions into a single coherent paragraph that describes the image.'''
                    
                    # Build full prompt (DeepSeek format)
                    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
                    
                    prompts.append(prompt)
                    batch_info.append({
                        "index": batch_start + i,
                        "image_id": image_id,
                        "image_path": image_path,
                        "dataset": dataset,
                        "captions": captions,
                        "prompt": prompt,
                        "retries": 0
                    })
                except Exception as e:
                    error_msg = f"Exception during prompt preparation: {str(e)}\n{traceback.format_exc()}"
                    log_error(error_msg, image_id, batch_start + i + 1)
            
            # Track items that need retries
            retry_indices = list(range(len(prompts)))
            retry_prompts = prompts.copy()
            
            # Process batch with retry logic
            while retry_indices and any(batch_info[i]["retries"] < MAX_RETRIES for i in retry_indices):
                try:
                    if retry_prompts:
                        outputs = llm.generate(retry_prompts, sampling_params=sampling_params)
                        
                        new_retry_indices = []
                        new_retry_prompts = []
                        
                        # Process each output
                        for i, output_idx in enumerate(retry_indices):
                            info = batch_info[output_idx]
                            output = outputs[i]
                            idx = info["index"]
                            image_id = info["image_id"]
                            
                            try:
                                raw_response = output.outputs[0].text.strip()
                                description = extract_description(raw_response)
                                
                                # Validate response
                                is_valid, error_reason = validate_description(description)
                                
                                if is_valid:
                                    # Create output item
                                    result_item = {
                                        "image": info["image_path"],
                                        "description": description,
                                        "image_id": image_id,
                                        "dataset": info["dataset"]
                                    }
                                    
                                    # Check if completed after retries
                                    if info["retries"] > 0:
                                        status_msg = f"(after {info['retries']} retries)"
                                    else:
                                        status_msg = ""
                                    
                                    results.append(result_item)
                                    processed_image_ids.append(image_id)
                                    
                                else:
                                    # Need to retry
                                    info["retries"] += 1
                                    if info["retries"] < MAX_RETRIES:
                                        new_retry_indices.append(output_idx)
                                        new_retry_prompts.append(info["prompt"])
                                    else:
                                        error_msg = f"Max retries reached. Last error: {error_reason}. Last raw response: {raw_response}"
                                        log_error(error_msg, image_id, idx+1)
                                
                            except Exception as e:
                                error_msg = f"Exception processing output: {str(e)}\n{traceback.format_exc()}"
                                log_error(error_msg, image_id, idx+1)
                                
                                # Try to retry
                                info["retries"] += 1
                                if info["retries"] < MAX_RETRIES:
                                    new_retry_indices.append(output_idx)
                                    new_retry_prompts.append(info["prompt"])
                        
                        # Update retry queue
                        retry_indices = new_retry_indices
                        retry_prompts = new_retry_prompts
                        
                except Exception as e:
                    error_msg = f"Batch generation exception: {str(e)}\n{traceback.format_exc()}"
                    print(f"Error in batch generation: {error_msg}")
                    
                    # Increment retry count for all items in batch on exception
                    new_retry_indices = []
                    new_retry_prompts = []
                    
                    for i, output_idx in enumerate(retry_indices):
                        info = batch_info[output_idx]
                        info["retries"] += 1
                        if info["retries"] < MAX_RETRIES:
                            new_retry_indices.append(output_idx)
                            new_retry_prompts.append(info["prompt"])
                        else:
                            log_error(error_msg, info["image_id"], info["index"]+1)
                    
                    retry_indices = new_retry_indices
                    retry_prompts = new_retry_prompts
            
            # Update progress bar
            pbar.update(len(batch_image_ids))
            
            # Save checkpoint periodically
            processed_count = len(processed_image_ids)
            if processed_count % checkpoint_interval == 0 or batch_end >= total_items:
                print(f"\nSaving checkpoint at {processed_count} processed items...")
                save_checkpoint(results, processed_count, processed_image_ids, output_path)
    
    # Save final results
    try:
        ensure_directory_exists(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
        print(f"Generated {len(results)} descriptions. Saved to {output_path}")
    except Exception as e:
        print(f"Error saving final results: {str(e)}")
        traceback.print_exc()
        
        # Try saving to backup location
        backup_path = f"{output_path}.backup.{int(time.time())}.json"
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False)
            print(f"Saved backup results to: {backup_path}")
        except Exception as backup_error:
            print(f"Failed to save backup: {str(backup_error)}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate combined image descriptions using DeepSeek model")
    parser.add_argument("--input_path", type=str, default=INPUT_PATH, help="Path to the input JSON file")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Output path for generated data")
    parser.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL, help="Checkpoint interval")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES, help="Maximum number of retries for failed generations")
    args = parser.parse_args()
    
    # Update max retries
    MAX_RETRIES = args.max_retries
    
    # Set up signal handlers
    setup_signal_handlers()
    
    # Ensure output directory exists
    ensure_directory_exists(args.output_path)
    
    try:
        # Process data and generate descriptions
        results = process_descriptions(
            args.input_path,
            args.output_path,
            args.checkpoint_interval,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size
        )
        
        print(f"Successfully generated {len(results)} descriptions. Saved to {args.output_path}")
    except KeyboardInterrupt:
        print("Program safely interrupted")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()