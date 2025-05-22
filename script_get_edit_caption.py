import os
import json
import argparse
import time
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

# Define type descriptions
TYPE_DESCRIPTIONS = {
    "Object": "Adding or removing entities (people, animals, food, vehicles, objects)",
    "Attribute": "Changing visual properties (color, material, size, shape)",
    "Scene": "Modifying backgrounds or settings",
    "Spatial Relation": "Altering physical arrangements (right, left, on top, below, inside)",
    "Action Relation": "Changing interactions between entities",
    "Part Relation": "Modifying part-whole relationships (body parts, clothing, accessories)",
    "Counting": "Adjusting quantity of entities",
    "Differentiation": "Distinguishing objects by attributes",
    "Comparison": "Changing relative characteristics",
    "Negation": "Removing elements",
    "Universality": "Applying changes to all members of a group"
}

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
        temperature=0.7,
        top_p=0.9,
        max_tokens=300,  # Longer text for caption generation
    )
    
    return llm, processor, sampling_params

def generate_new_caption(edited_image_path, original_description, type_value, 
                          llm, processor, sampling_params, max_retries=3):
    """Generate a new caption based on the provided image and description"""
    
    try:
        # Open image and ensure it exists
        img = Image.open(edited_image_path)
    except Exception as e:
        return f"Error: Unable to load image {edited_image_path}: {str(e)}"
    
    # Process type value, ensure correct format
    type_value = type_value.strip().replace("**", "")
    
    # Get type description
    type_description = TYPE_DESCRIPTIONS.get(type_value, "")
    
    # Construct prompt
    prompt_template = '''
    Your task is generating an accurate caption for the given image, using the following information as context:

    Reference description: {description}
    Key aspect to focus on: {type} ({type_description})

    Your task is to:
    1. Carefully examine the actual image content
    2. Note any differences between the image and the reference description, especially regarding the {type} aspects
    3. Create an accurate caption that accurately describes ONLY what is truly visible in the image

    IMPORTANT:
    - Your description must be based solely on what is actually visible in the image
    - Do not include details from the reference description that aren't present in the image
    - Pay special attention to accurately describing the {type} elements
    - Be precise and objective in your description

    Please provide your accurate caption in English, beginning with "CAPTION:".'''

    prompt = prompt_template.format(
        description=original_description,
        type=type_value,
        type_description=type_description
    )
    
    # Prepare messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "image": edited_image_path,
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
    
    # Process vision information - directly use PIL image object
    image_inputs = [img]
    video_inputs = None
    video_kwargs = {}
    
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
            return response_text
        except Exception as e:
            print(f"Retry {retry+1}/{max_retries} failed: {str(e)}")
            time.sleep(2)  # Wait longer after errors
    
    # If all retries fail, return error
    return f"Error: All retries failed"

def log_error(error_msg, item_id, error_file="./logs/caption_error.txt"):
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
    # Ensure correct data format
    if "results" not in data:
        print("Error: Input JSON does not contain 'results' field")
        return []
    
    # If shard is specified, only process that shard
    if shard_id is not None and total_shards is not None:
        shard_size = len(data["results"]) // total_shards
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size if shard_id < total_shards - 1 else len(data["results"])
        data_shard = data["results"][start_idx:end_idx]
        print(f"Processing shard {shard_id+1}/{total_shards}, {len(data_shard)} items (total data: {len(data['results'])} items)")
    else:
        data_shard = data["results"]
        print(f"Processing complete dataset, {len(data_shard)} items")
    
    # Load checkpoint
    results, start_idx = load_checkpoint(output_path, shard_id)
    
    if start_idx > 0:
        print(f"Resuming from checkpoint: {start_idx} items already processed")
    
    # Initialize model
    llm, processor, sampling_params = initialize_model(tensor_parallel_size=tensor_parallel_size)
    
    # Track success/failure counts
    success_count = 0
    error_count = 0
    
    # Process each image
    for idx, item in enumerate(tqdm(data_shard[start_idx:], initial=start_idx, total=len(data_shard)), start=start_idx):
        edited_image_name = item["edited_image_name"]
        
        # Choose correct description field based on data structure
        if "complete_caption" in item and item["complete_caption"]:
            original_description = item["complete_caption"]
        elif "original_description" in item and item["original_description"]:
            original_description = item["original_description"]
        else:
            print(f"Warning: Item {idx} has no valid description field")
            original_description = ""
            
        type_value = item["type"]
        
        # If edited image file exists
        if os.path.exists(edited_image_name):
            try:
                # Generate new caption
                new_caption = generate_new_caption(
                    edited_image_name,
                    original_description,
                    type_value,
                    llm,
                    processor,
                    sampling_params
                )
                
                # Save result
                result_item = {
                    **item,  # Keep original data
                    "new_caption": new_caption,
                }
                
                # Check if generation was successful
                if not new_caption.startswith("Error:"):
                    success_count += 1
                else:
                    error_count += 1
                
                # Add to results list
                results.append(result_item)
                
            except Exception as e:
                error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
                log_error(error_msg, edited_image_name)
                
                # Add error result
                result_item = {
                    **item,
                    "new_caption": "Error",
                    "error_message": str(e),
                }
                
                error_count += 1
                results.append(result_item)
        else:
            print(f"Warning: Edited image does not exist: {edited_image_name}")
            # Add error result
            result_item = {
                **item,
                "new_caption": "Error",
                "error_message": "Edited image not found",
            }
            
            error_count += 1
            results.append(result_item)
        
        # Save checkpoint periodically
        if (idx + 1) % checkpoint_interval == 0:
            save_checkpoint(results, idx + 1, output_path, shard_id)
    
    # Save final results
    suffix = f"_shard{shard_id}" if shard_id is not None else ""
    final_output_path = f"{os.path.splitext(output_path)[0]}{suffix}.json"
    
    final_data = {
        "processed_count": len(data_shard),
        "results": results
    }
    
    with open(final_output_path, 'w') as f:
        json.dump(final_data, f, indent=4)
    
    # Also save final checkpoint
    save_checkpoint(results, len(data_shard), output_path, shard_id)
    
    print(f"Completed processing shard {shard_id if shard_id is not None else 'all'}")
    print(f"Results saved to: {final_output_path}")
    print(f"Successful samples: {success_count}, Error samples: {error_count}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new captions for edited images")
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
    data = load_json_data(args.input_json)
    
    # Process specified shard
    process_data_shard(
        data,
        args.output_path,
        args.shard_id,
        args.total_shards,
        args.checkpoint_interval,
        args.tensor_parallel_size
    )