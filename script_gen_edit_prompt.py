import os
import json
import re
import traceback
from pathlib import Path
from datetime import datetime

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

from utils.utils import encode_image, get_edit_base_prompt

# Model path for locally deployed model
MODEL_PATH = "path/to/Qwen2.5-VL-72B-Instruct"

def load_filtered_images(json_path):
    """Load filtered image data from JSON file."""
    with open(json_path, 'r') as f:
        filtered_data = json.load(f)
    return filtered_data

def extract_type_and_prompt(response_text):
    """Extract composition type and prompt from model response."""
    type_match = re.search(r'## Type: (.*?)(?:\n|$)', response_text)
    prompt_match = re.search(r'## Prompt: (.*?)(?:\n|$)', response_text)
    
    composition_type = type_match.group(1).strip() if type_match else "Unknown"
    prompt = prompt_match.group(1).strip() if prompt_match else response_text.strip()
    
    return composition_type, prompt

def log_error(error_msg, image_id, idx, error_file="./logs/vg_error.txt"):
    """Log error information to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_file, 'a') as f:
        f.write(f"[{timestamp}] Error processing image {idx}: {image_id}\n")
        f.write(f"Error message: {error_msg}\n")
        f.write("-" * 80 + "\n")
    print(f"Error logged to {error_file}")

def initialize_model(model_path=MODEL_PATH, tensor_parallel_size=4, max_model_len=128000, batch_size=4):
    """Initialize local Qwen model with vLLM."""
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.95,
        max_num_seqs=batch_size,
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_prefix_caching=True,
    )
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=1024,
        stop_token_ids=[],
    )
    
    return llm, processor, sampling_params

def gen_edit_prompt(image_path, image_caption=None, llm=None, processor=None, sampling_params=None, max_image_pixels=2056*2056):
    """Generate edit prompt using local Qwen model."""
    if llm is None or processor is None or sampling_params is None:
        llm, processor, sampling_params = initialize_model()
    
    content = get_edit_base_prompt().format(image_caption if image_caption else "")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": content},
            {
                "type": "image",
                "image": image_path,
                "min_pixels": 224 * 224,
                "max_pixels": max_image_pixels,
            }
        ]}
    ]
    
    # Process message with model
    prompt = processor.apply_chat_template(
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
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    # Generate response
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    response_text = outputs[0].outputs[0].text
    
    # Extract composition type and prompt
    composition_type, prompt = extract_type_and_prompt(response_text)
    
    return composition_type, prompt, response_text

def find_original_image_path(image_path, base_dirs):
    """
    Find original image path by mapping dataset path to actual path.
    For VG dataset, extract filename and search in base_dirs.
    """
    # Extract filename (e.g., "1.jpg" from "/export/share/datasets/vision/visual-genome/image/1.jpg")
    image_name = os.path.basename(image_path)
    
    # First try direct lookup in base_dirs
    for base_dir in base_dirs:
        potential_path = os.path.join(base_dir, image_name)
        if os.path.exists(potential_path):
            return potential_path
    
    # If direct lookup fails, do recursive search
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            if image_name in files:
                return os.path.join(root, image_name)
    
    print(f"Could not find image {image_name} in any of the base directories")
    return None

def save_checkpoint(results, processed_count, output_path):
    """Save checkpoint with progress information."""
    checkpoint_path = f"{os.path.splitext(output_path)[0]}_vg_checkpoint.json"
    checkpoint_data = {
        "processed_count": processed_count,
        "results": results
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=4)
    print(f"Checkpoint saved: {checkpoint_path} (Processed: {processed_count} items)")
    return checkpoint_path

def load_checkpoint(output_path):
    """Load checkpoint if it exists."""
    checkpoint_path = f"{os.path.splitext(output_path)[0]}_vg_checkpoint.json"
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data["results"], checkpoint_data["processed_count"]
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
    return [], 0

def process_filtered_images(filtered_data, base_dirs, output_path, checkpoint_interval=10, batch_size=4, tensor_parallel_size=4):
    """Process filtered images, generate prompts and save results with checkpoints."""
    # Check for existing checkpoint
    results, start_idx = load_checkpoint(output_path)
    
    if start_idx > 0:
        print(f"Resuming from checkpoint: {start_idx} items already processed.")
    
    # Initialize model
    print(f"Initializing local model from {MODEL_PATH} with {tensor_parallel_size} GPUs...")
    llm, processor, sampling_params = initialize_model(
        model_path=MODEL_PATH,
        tensor_parallel_size=tensor_parallel_size,
        batch_size=batch_size
    )
    
    # Process images
    for idx, item in enumerate(filtered_data[start_idx:], start=start_idx):
        image_path = item["image"]
        description = item["description"]
        image_id = item["image_id"]
        dataset = item.get("dataset", "vg")
        
        # Find original image path
        actual_image_path = find_original_image_path(image_path, base_dirs)
        
        if actual_image_path:
            print(f"Processing {image_id} ({idx+1}/{len(filtered_data)})...")
            try:
                composition_type, edit_prompt, full_response = gen_edit_prompt(
                    actual_image_path, 
                    description, 
                    llm=llm, 
                    processor=processor, 
                    sampling_params=sampling_params
                )
                
                # Save result with type field
                result = {
                    "image": image_path,
                    "image_id": image_id,
                    "dataset": dataset,
                    "description": description,
                    "edit_prompt": edit_prompt,
                    "type": composition_type,
                    "full_response": full_response
                }
                results.append(result)
                
            except Exception as e:
                # Log exception
                error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
                log_error(error_msg, image_id, idx+1)
                
                # Add placeholder result
                result = {
                    "image": image_path,
                    "image_id": image_id,
                    "dataset": dataset,
                    "description": description,
                    "edit_prompt": f"ERROR: {type(e).__name__}",
                    "type": "Error",
                    "full_response": str(e),
                    "error": True
                }
                results.append(result)
                continue
            
            # Save checkpoint periodically
            if (idx + 1) % checkpoint_interval == 0:
                save_checkpoint(results, idx + 1, output_path)
                
        else:
            print(f"Could not find original path for {image_path}")
            log_error("Image path not found", image_id, idx+1)
    
    # Save final results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save final checkpoint
    save_checkpoint(results, len(filtered_data), output_path)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate edit prompts for filtered images")
    parser.add_argument("--filtered_json_path", type=str, default="/path/to/datasets/vg_descriptions_split2.json", help="Path to filtered images JSON file")
    parser.add_argument("--base_dirs", type=str, nargs="+", default=["/path/to/Data/visualgenome/VG_100K"], help="Base directories to search for original images")
    parser.add_argument("--output_path", type=str, default="datasets/generated_edit_vg_prompts.json", help="Output path for generated prompts")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Checkpoint interval (save every N items)")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    args = parser.parse_args()
    
    # Load filtered image data
    filtered_data = load_filtered_images(args.filtered_json_path)
    
    # Process filtered images with checkpoints
    results = process_filtered_images(
        filtered_data, 
        args.base_dirs, 
        args.output_path, 
        args.checkpoint_interval,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    print(f"Generated {len(results)} edit prompts. Saved to {args.output_path}")