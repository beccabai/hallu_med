import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate CLIP embedding cosine similarity between image pairs')
    parser.add_argument('--input_json', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output_json', type=str, drequired=True, help='Output JSON file path')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to CLIP model')
    parser.add_argument('--split_id', type=int, default=0, help='Split ID (0-based)')
    parser.add_argument('--num_splits', type=int, default=1, help='Total number of splits')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--ckpt_interval', type=int, default=10, help='Checkpoint interval (items)')
    return parser.parse_args()

def load_data(filepath, split_id, num_splits):
    """Load and split the data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract the results list from the data
    results_list = data["results"]
    
    # Calculate split indices
    total_items = len(results_list)
    items_per_split = total_items // num_splits
    start_idx = split_id * items_per_split
    
    # Handle the last split (may have extra items)
    if split_id == num_splits - 1:
        end_idx = total_items
    else:
        end_idx = start_idx + items_per_split
    
    return results_list[start_idx:end_idx], start_idx, end_idx

def load_checkpoint(output_path, split_id):
    """Load checkpoint if exists"""
    ckpt_path = f"{output_path}_split{split_id}_ckpt.json"
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r') as f:
            return json.load(f)
    return {"processed_items": 0, "results": []}

def save_checkpoint(output_path, split_id, processed_items, results):
    """Save checkpoint"""
    ckpt_path = f"{output_path}_split{split_id}_ckpt.json"
    with open(ckpt_path, 'w') as f:
        json.dump({"processed_items": processed_items, "results": results}, f, indent=2)

def save_final_results(output_path, results):
    """Save final results"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def calculate_cosine_similarity(model, processor, image1_path, image2_path, device):
    """Calculate cosine similarity between two images using CLIP embeddings"""
    try:
        # Load images
        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')
        
        # Process images
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        inputs2 = processor(images=image2, return_tensors="pt").to(device)
        
        # Get embeddings
        with torch.no_grad():
            vision_outputs1 = model.vision_model(**inputs1)
            image_embeds1 = vision_outputs1.pooler_output
            
            vision_outputs2 = model.vision_model(**inputs2)
            image_embeds2 = vision_outputs2.pooler_output
            
            # Normalize embeddings
            image_embeds1 = image_embeds1 / image_embeds1.norm(dim=-1, keepdim=True)
            image_embeds2 = image_embeds2 / image_embeds2.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = torch.matmul(image_embeds1, image_embeds2.T).item()
            
            return similarity
    except Exception as e:
        print(f"Error processing images {image1_path} and {image2_path}: {e}")
        return None

def process_in_batches(data, model, processor, start_from, args, device):
    """Process data in batches with checkpoint support"""
    results = []
    processed_items = start_from
    
    # Skip already processed items
    data_to_process = data[start_from:]
    
    for i, item in enumerate(tqdm(data_to_process, desc=f"Processing split {args.split_id}")):
        # Calculate current global index
        current_idx = start_from + i
        
        image_path = item["image_name"]
        edited_image_path = item["edited_image_name"]
        
        # Calculate CLIP similarity
        similarity = calculate_cosine_similarity(model, processor, image_path, edited_image_path, device)
        
        # Create result
        result = {
            "image_name": image_path,
            "edited_image_name": edited_image_path,
            "CLIP_Similarity": similarity
        }
        
        results.append(result)
        processed_items = current_idx + 1
        
        # Save checkpoint at intervals
        if (i + 1) % args.ckpt_interval == 0:
            save_checkpoint(args.output_json, args.split_id, processed_items, results)
            print(f"Checkpoint saved at item {processed_items}")
    
    return results

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading CLIP model from {args.model_path}")
    model = AutoModelForZeroShotImageClassification.from_pretrained(args.model_path).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Load data
    print(f"Loading data from {args.input_json}")
    data, start_idx, end_idx = load_data(args.input_json, args.split_id, args.num_splits)
    print(f"Processing split {args.split_id} of {args.num_splits} (items {start_idx} to {end_idx-1})")
    
    # Load checkpoint
    ckpt = load_checkpoint(args.output_json, args.split_id)
    start_from = ckpt["processed_items"]
    results = ckpt["results"]
    
    if start_from > 0:
        print(f"Resuming from checkpoint at item {start_from}")
    
    # Process data
    if start_from < len(data):
        new_results = process_in_batches(data, model, processor, start_from, args, device)
        results.extend(new_results)
    
    # Save final results
    output_filename = f"{args.output_json}_split{args.split_id}.json"
    save_final_results(output_filename, results)
    print(f"Results saved to {output_filename}")
    
    # Clean up checkpoint file
    ckpt_path = f"{args.output_json}_split{args.split_id}_ckpt.json"
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"Removed checkpoint file {ckpt_path}")

if __name__ == "__main__":
    main()