import os
import json
import argparse
import time
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import traceback

from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

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
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_prefix_caching=True,
    )
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=256,  # Allow longer descriptions
    )
    
    return llm, processor, sampling_params

def generate_complete_caption(image_path, original_description, edit_prompt, 
                            llm, processor, sampling_params, max_retries=3):
    """Generate complete image description"""
    
    # Build prompt
    prompt_template = """Given an image, its current caption, and an edit prompt, your task is to determine whether the caption fully incorporates all information from the edit prompt. If not, you need to update the caption to include any missing information.

Image: [The image will be provided]
Current Caption: {original_description}
Edit Prompt: {edit_prompt}

IMPORTANT: The complete caption should ONLY describe what is ACTUALLY VISIBLE in the image. DO NOT add information from the edit prompt if those elements don't actually exist in the image.

Follow these steps:
1. Carefully examine the image and understand the current caption
2. Identify all new or modified information requested in the edit prompt
3. Verify if each element from the edit prompt is actually visible in the image
4. Check whether the current caption already contains all relevant and visible information from the edit prompt
5. If the caption FULLY incorporates all relevant information from the edit prompt, respond with:
   "Caption is complete. No updates needed."
6. If the caption is MISSING any relevant information that is ACTUALLY VISIBLE in the image, create an updated caption by:
   - Preserving relevant parts of the original caption
   - Naturally integrating ONLY the missing information that is actually visible in the image
   - Maintaining a coherent, fluent, and accurate description
   - Ensuring the updated caption accurately describes what is visible in the image
   - EXCLUDING any information from the edit prompt that is not actually present in the image

Provide only the updated caption without any explanations, or confirm that no updates are needed (By Answer exactly "NO UPDATE").
"""

    prompt = prompt_template.format(
        original_description=original_description,
        edit_prompt=edit_prompt
    )
    
    # Prepare messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "image": image_path,
                "min_pixels": 224 * 224,
                "max_pixels": 2048 * 2048,
            }
        ]}
    ]
    
    # Process messages with model's chat template
    formatted_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Process vision information
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    llm_inputs = {
        "prompt": formatted_prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    # Generation with retry mechanism
    for retry in range(max_retries):
        try:
            outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            
            # Check if update needed
            if response_text.lower() == "no update" or "caption is complete" in response_text.lower():
                return original_description, "NO UPDATE"
            else:
                return response_text, "UPDATED"
        except Exception as e:
            print(f"Retry {retry+1}/{max_retries} failed: {str(e)}")
            time.sleep(2)  # Wait longer after errors
    
    # If all retries fail, return original description
    return original_description, f"ERROR: All retries failed, last response: {response_text if 'response_text' in locals() else 'No response'}"

def verify_caption_match(image_path, complete_caption, 
                        llm, processor, sampling_params, max_retries=3):
    """Verify if generated description matches the image"""
    
    # Build prompt
    prompt_template = """Examine the provided image and the caption. Your task is to verify whether the caption accurately describes ONLY what is VISIBLE in the image.

Image: [Image is provided]
Caption: {complete_caption}

IMPORTANT INSTRUCTIONS:
1. Focus on what is ACTUALLY VISIBLE in the image
2. Check if the caption mentions objects, people, or elements that do NOT exist in the image
3. Check if the caption describes accurate attributes (colors, positions, actions, etc.) for what is visible
4. Verify if the caption matches the actual content of the image

Does this caption accurately describe ONLY what is visible in the image?
Answer with "Yes" if the caption is completely accurate. 
Answer with "No" followed by a brief explanation if the caption contains inaccuracies or mentions elements not present in the image.
"""

    prompt = prompt_template.format(
        complete_caption=complete_caption
    )
    
    # Prepare messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "image": image_path,
                "min_pixels": 224 * 224,
                "max_pixels": 2048 * 2048,
            }
        ]}
    ]
    
    # Process messages with model's chat template
    formatted_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Process vision information
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    llm_inputs = {
        "prompt": formatted_prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    # Set verification sampling parameters
    verify_sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=256,  # Allow brief explanations
    )
    
    # Generation with retry mechanism
    for retry in range(max_retries):
        try:
            outputs = llm.generate([llm_inputs], sampling_params=verify_sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            
            # Extract Yes or No
            if response_text.lower().startswith("yes"):
                return "Yes"
            elif response_text.lower().startswith("no"):
                # If No, we also return the reason
                return response_text
            else:
                # Try to extract Yes or No from the response
                if "yes" in response_text.lower():
                    return "Yes"
                elif "no" in response_text.lower():
                    return response_text
                else:
                    print(f"Retry {retry+1}/{max_retries}: Incorrect output format: '{response_text}'")
                    time.sleep(1)
        except Exception as e:
            print(f"Retry {retry+1}/{max_retries} failed: {str(e)}")
            time.sleep(2)
    
    # If all retries fail, return error
    return f"Error: Unable to determine if description matches image"

def log_error(error_msg, item_id, error_file="./logs/generate_error.txt"):
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
    llm, processor, sampling_params = initialize_model(tensor_parallel_size=tensor_parallel_size)
    
    # Process each image
    for idx, item in enumerate(tqdm(data_shard[start_idx:], initial=start_idx, total=len(data_shard)), start=start_idx):
        image_name = item["image_name"]
        original_description = item["original_description"]
        edit_prompt = item["edit_prompt"]
        
        # Create new result item, preserve original data but remove verification_result
        result_item = {k: v for k, v in item.items() if k != "verification_result"}
        
        # If image file exists
        if os.path.exists(image_name):
            try:
                # 1. Generate complete description
                complete_caption, update_status = generate_complete_caption(
                    image_name,
                    original_description,
                    edit_prompt,
                    llm,
                    processor,
                    sampling_params
                )
                
                # Add complete description to result
                result_item["complete_caption"] = complete_caption
                
                # 2. Verify if description matches image
                # If first inference result is "NO UPDATE", use original description for verification
                description_to_verify = original_description if update_status == "NO UPDATE" else complete_caption
                
                caption_match_result = verify_caption_match(
                    image_name,
                    description_to_verify,
                    llm,
                    processor,
                    sampling_params
                )
                
                # Add verification result
                result_item["complete_caption_judge"] = caption_match_result
                
            except Exception as e:
                error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
                log_error(error_msg, image_name)
                
                # Add error information
                result_item["complete_caption"] = "Error: " + str(e)
                result_item["complete_caption_judge"] = "Error"
        else:
            print(f"Warning: Image does not exist: {image_name}")
            # Add error information
            result_item["complete_caption"] = "Error: Image not found"
            result_item["complete_caption_judge"] = "Error"
        
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
    parser = argparse.ArgumentParser(description="Generate complete image descriptions and verify")
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON data path")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint interval (save every N items)")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID to process (starting from 0)")
    parser.add_argument("--total_shards", type=int, default=8, help="Total number of shards")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load data
    data = load_json_data(args.input_json)["results"] if "results" in load_json_data(args.input_json) else load_json_data(args.input_json)
    
    # Process specified shard
    process_data_shard(
        data,
        args.output_path,
        args.shard_id,
        args.total_shards,
        args.checkpoint_interval,
        args.tensor_parallel_size
    )