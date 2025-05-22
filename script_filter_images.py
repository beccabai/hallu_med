import os
import json
import argparse
import base64
import math
from tqdm import tqdm
from openai import OpenAI
from utils.utils import encode_image
from utils.utils import get_filter_prompt

def gen_filter_prompt(image_path, image_caption=None):
    """Generate filtering prompt using Alibaba Cloud Bailian API with Qwen2.5-VL"""
    client = OpenAI(
        api_key=os.getenv("ALICLOUD_BAILIAN_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    base64_image = encode_image(image_path)
    if image_caption:
        content = get_filter_prompt().format(caption=image_caption)
    else:
        content = get_filter_prompt().format(caption="No caption provided")
    
    completion = client.chat.completions.create(
        model="qwen2.5-vl-32b-instruct",
        messages=[{"role": "user","content": [
                {"type": "text","text": content},
                {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}]
        )
    prompt = completion.choices[0].message.content
    
    return prompt

def load_dataset(base_dir="docci", images_dir="docci/images", limit=None):
    descriptions_path = os.path.join(base_dir, 'docci_descriptions.jsonlines')
    
    samples = []
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and len(samples) >= limit:
                break
                
            data = json.loads(line.strip())
            
            image_file = data['image_file']
            image_path = os.path.join(images_dir, image_file)
            
            if os.path.exists(image_path):
                samples.append({
                    'image_path': image_path,
                    'description': data['description'],
                    'image_name': image_file,
                    'example_id': data['example_id']
                })
    
    return samples

def contains_yes(text):
    """
    Check if the Final Recommendation in the text is 'Yes'
    Looks for various common formats of 'Yes' recommendation
    """
    import re
    
    # Try several patterns to find the final recommendation
    patterns = [
        r"Final Recommendation \(Yes/No\):\s*(Yes|No)",
        r"Final Recommendation:\s*(?:\*\*)?(?:Yes|No)(?:\*\*)?",
        r"### Final Recommendation:\s*(?:\*\*)?(?:Yes|No)(?:\*\*)?",
    ]
    
    # Check if text contains markdown-formatted "Yes" recommendation
    if re.search(r"Final Recommendation:.*\*\*Yes\*\*", text, re.DOTALL) or \
       re.search(r"### Final Recommendation:.*\*\*Yes\*\*", text, re.DOTALL):
        return True
    
    # Check for plain text version
    if "Final Recommendation: Yes" in text or \
       "Final Recommendation (Yes/No): Yes" in text:
        return True
    
    # More comprehensive check for various formats
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and "yes" in match.group(0).lower():
            return True
    
    return False

def ensure_dir_exists(file_path):
    """Ensure file directory exists, creating all intermediate directories"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def split_samples(samples, total_parts, part_index):
    """Split samples into N parts and return the nth part"""
    if part_index < 1 or part_index > total_parts:
        raise ValueError(f"Part index must be between 1 and {total_parts}")
    
    total_samples = len(samples)
    part_size = math.ceil(total_samples / total_parts)
    
    start_idx = (part_index - 1) * part_size
    end_idx = min(start_idx + part_size, total_samples)
    
    return samples[start_idx:end_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter DOCCI dataset images")
    parser.add_argument("--base_dir", type=str, default="../docci", help="Base directory containing DOCCI files")
    parser.add_argument("--images_dir", type=str, default="../docci/images", help="Directory containing DOCCI images")
    parser.add_argument("--output_dir", type=str, default="datasets", help="Directory to save output files")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process")
    parser.add_argument("--total_parts", type=int, default=1, help="Total number of parts to split the dataset")
    parser.add_argument("--part_index", type=int, default=1, help="Index of the part to process (1-based)")
    args = parser.parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")
    
    # Build output filenames (with part information)
    filtered_output = os.path.join(args.output_dir, f"filtered_image_docci_part_{args.part_index}.json")
    debug_output = os.path.join(args.output_dir, f"debug_image_docci_part_{args.part_index}.json")
    
    # 1. Load DOCCI dataset images and descriptions
    print(f"Loading DOCCI dataset from {args.base_dir}...")
    all_samples = load_dataset(base_dir=args.base_dir, images_dir=args.images_dir, limit=args.limit)
    print(f"Loaded {len(all_samples)} total samples")
    
    # 2. Split samples and get current part
    try:
        part_samples = split_samples(all_samples, args.total_parts, args.part_index)
        print(f"Processing part {args.part_index} of {args.total_parts} with {len(part_samples)} samples")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Create result lists
    filtered_results = []  # Samples that passed filtering
    debug_results = []     # Samples that didn't pass filtering
    
    # 3. Process each sample and save results
    print("Processing samples...")
    processed_count = 0
    filtered_count = 0
    debug_count = 0
    
    for sample in tqdm(part_samples):
        try:
            processed_count += 1
            
            # Call model to analyze image and description
            reason = gen_filter_prompt(sample['image_path'], sample['description'])
            
            # Create result object
            result = {
                "image_name": sample['image_name'],
                "description": sample['description'],
                "reason": reason
            }
            
            # Save to different files based on whether it contains "Yes"
            if contains_yes(reason):
                filtered_results.append(result)
                filtered_count += 1
                
                # Save checkpoint every 5 filtered samples
                if filtered_count % 5 == 0:
                    with open(filtered_output, 'w', encoding='utf-8') as f:
                        json.dump(filtered_results, f, ensure_ascii=False, indent=4)
                    print(f"Checkpoint saved: {filtered_count} filtered results")
            else:
                debug_results.append(result)
                debug_count += 1
                
                # Save checkpoint every 5 debug samples
                if debug_count % 5 == 0:
                    with open(debug_output, 'w', encoding='utf-8') as f:
                        json.dump(debug_results, f, ensure_ascii=False, indent=4)
                    print(f"Checkpoint saved: {debug_count} debug results")
            
        except Exception as e:
            print(f"Error processing {sample['image_name']}: {e}")
            with open(os.path.join(args.output_dir, f"errors_part_{args.part_index}.txt"), 'a') as f:
                f.write(f"{sample['image_name']}: {str(e)}\n")
    
    # Save final results
    print(f"Saving filtered results to {filtered_output}...")
    with open(filtered_output, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=4)
    
    print(f"Saving debug results to {debug_output}...")
    with open(debug_output, 'w', encoding='utf-8') as f:
        json.dump(debug_results, f, ensure_ascii=False, indent=4)
    
    print(f"Summary for part {args.part_index} of {args.total_parts}:")
    print(f"- Total samples in this part: {len(part_samples)}")
    print(f"- Processed: {processed_count} samples")
    print(f"- Filtered (with 'Yes'): {filtered_count} samples")
    print(f"- Debug (without 'Yes'): {debug_count} samples")