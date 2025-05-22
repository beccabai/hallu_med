import json
import os
import argparse
import re
from pathlib import Path
import torch
import random
from typing import Optional
from collections import defaultdict
from vllm import LLM, EngineArgs, SamplingParams
from vllm.multimodal.utils import fetch_image
from PIL import Image

def initialize_model(model_path, lora_path=None):
    """Initialize Llama-3.2-Vision model using vLLM"""
    print(f"Initializing model: {model_path}")
    
    # Configure the model
    engine_args = EngineArgs(
        model=model_path,
        max_model_len=1024,
        max_num_seqs=16,
        limit_mm_per_prompt={"image": 2},  # Set to 2 for our two-image benchmarks
    )
    
    # Initialize the LLM
    llm = LLM(**engine_args.__dict__)
    
    print("Model initialization complete")
    return llm

def ask_question_with_images(llm, image_paths, question, options=None):
    """Use Llama-3.2-Vision model to answer questions about two images"""
    
    # Build prompt with images, explicitly requesting output format
    content = "Please carefully observe these two images and answer the following question:\n\n" + question
    
    # Add options to the prompt if available
    if options:
        content += "\n\nOptions:\n"
        for option in options:
            content += f"{option}\n"
        content += "\nPlease answer strictly in the following format: start with 'Answer:', then give the letter of the correct option (without extra punctuation or spaces), For example, 'Answer: A'."
    
    # Format the prompt with image placeholders
    placeholders = "<|image|>" * len(image_paths)
    prompt = f"{placeholders}<|begin_of_text|>{content}"
    
    # Load images from file paths
    image_data = []
    for img_path in image_paths:
        # Open the image file
        img = Image.open(img_path)
        # Convert to RGB if it's not already
        if img.mode != "RGB":
            img = img.convert("RGB")
        image_data.append(img)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.5,
        max_tokens=20,
    )
    
    # Generate the response
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data
            },
        },
        sampling_params=sampling_params,
    )
    
    # Extract the generated text
    response_text = outputs[0].outputs[0].text
    
    return response_text

def load_samples(json_path, num_samples=100):
    """Load sample data from JSON file and randomly select specified number of samples"""
    with open(json_path, 'r') as f:
        data = json.load(f)
        all_samples = data.get("results", [])
    
    # Return all samples if total count is less than requested
    if len(all_samples) <= num_samples:
        return all_samples
    
    # Otherwise randomly select specified number of samples
    selected_samples = random.sample(all_samples, num_samples)
    print(f"Randomly selected {num_samples} samples from {len(all_samples)} total samples")
    return selected_samples

def extract_answer(response, options=None):
    """Extract option letter from model response, randomly select one if invalid format"""
    # Use regex to strictly match "Answer: X" format, where X is a letter
    match = re.search(r"Answer:\s*([A-Za-z])", response)
    if match:
        return match.group(1).upper()  # Return uppercase letter
    
    # If no match found, randomly select from available options
    if options:
        possible_answers = [option[0].upper() for option in options]  # Get first letter of each option
        random_answer = random.choice(possible_answers)
        print(f"Warning: Could not extract answer format. Randomly selecting: {random_answer}")
        return random_answer
    
    # If no options provided, randomly select A-D
    random_answer = random.choice(['A', 'B', 'C', 'D'])
    print(f"Warning: Could not extract answer format. Randomly selecting: {random_answer}")
    return random_answer

def process_samples(samples, llm, output_file=None):
    """Process all samples and return results"""
    results = []
    
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}")
        
        question = sample["question"]
        options = sample["options"]
        answer = sample.get("answer", None)  # Get correct answer (if available)
        
        # Extract image paths from new JSON format
        image_paths = [
            sample["image_name"],
            sample["edited_image_name"]
        ]
        
        # Ensure image paths exist
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                print(f"Warning: Image path does not exist: {path}")
        
        if len(valid_paths) < 2:
            print(f"Error: Sample {i+1} does not have enough valid image paths")
            continue
        
        # Use model to answer the question
        response = ask_question_with_images(llm, valid_paths, question, options)
        
        # Extract model's answer option, passing options for random selection if needed
        extracted_answer = extract_answer(response, options)
        
        # Save results
        result = {
            "sample_id": i,
            "question": question,
            "options": options,
            "model_response": response,
            "extracted_answer": extracted_answer,
            "ground_truth": answer,
            "type": sample.get("type", "Unknown")  # Add the type field
        }
        results.append(result)
        
        print(f"Question: {question}")
        print(f"Model response: {response}")
        print(f"Extracted answer: {extracted_answer}")
        if answer:
            print(f"Correct answer: {answer}")
        print("-" * 50)
    
    # Save results to file (if specified)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    
    return results

def evaluate_results(results):
    """Evaluate model answer accuracy overall and by type"""
    correct = 0
    total = 0
    
    # Use defaultdict to track type-specific metrics
    type_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result in results:
        if not result.get("ground_truth"):
            continue
            
        ground_truth = result["ground_truth"].upper()  # Convert to uppercase for comparison
        extracted_answer = result.get("extracted_answer")
        result_type = result.get("type", "Unknown")
            
        # Overall accuracy
        if extracted_answer == ground_truth:
            correct += 1
        total += 1
        
        # Type-specific accuracy
        type_metrics[result_type]["total"] += 1
        if extracted_answer == ground_truth:
            type_metrics[result_type]["correct"] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct / total if total > 0 else 0
    
    # Calculate type-specific accuracies
    for type_name, metrics in type_metrics.items():
        type_metrics[type_name]["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
    
    # Print results
    print(f"Overall Accuracy: {correct}/{total} = {overall_accuracy:.2%}")
    print("\nAccuracy by Type:")
    for type_name, metrics in sorted(type_metrics.items()):
        print(f"  {type_name}: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.2%}")
    
    return {
        "overall": {
            "accuracy": overall_accuracy,
            "correct": correct,
            "total": total
        },
        "by_type": {type_name: metrics for type_name, metrics in type_metrics.items()}
    }

def save_incorrect_samples(results, original_samples, output_file):
    """Save samples that model answered incorrectly to a JSON file"""
    incorrect_samples = []
    
    for result in results:
        sample_id = result['sample_id']
        ground_truth = result.get('ground_truth')
        extracted_answer = result.get('extracted_answer')
        
        # Add sample to incorrect list if answer doesn't match ground truth
        if ground_truth and extracted_answer != ground_truth.upper():
            # Find the original sample
            original_sample = original_samples[sample_id]
            incorrect_samples.append(original_sample)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(incorrect_samples)} incorrect samples to {output_file}")
    return incorrect_samples

def main():
    parser = argparse.ArgumentParser(description="Use Llama-3.2-Vision model to answer questions about images")
    parser.add_argument("--model_path", type=str, 
                        required=True,
                        help="Path to the model")
    parser.add_argument("--json_path", type=str, 
                        default="./MED_bench.json",
                        help="Path to sample JSON file")
    parser.add_argument("--output_file", type=str, 
                        default='./final_output/llama3_2_v_results.json',
                        help="Path to output results file (will get timestamp appended)")
    parser.add_argument("--incorrect_file", type=str, 
                        default='./output/llama3_2_v_incorrect',
                        help="Path to save incorrect samples (will get timestamp & .json appended)")
    parser.add_argument("--num_samples", type=int, 
                        default=2000,
                        help="Number of samples to process")
    parser.add_argument("--seed", type=int, 
                        default=42,
                        help="Random seed for sample selection")
    args = parser.parse_args()
    
    # For model name in output files
    MODEL_NAME = "llama32_vision"

    # Load samples
    all_samples = load_samples(args.json_path, args.num_samples)
    print(f"Using model: {args.model_path}")
    print(f"Ready to process {len(all_samples)} samples")

    # Initialize model
    llm = initialize_model(model_path=args.model_path)

    # Generate timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build timestamped filenames
    base_out = os.path.splitext(args.output_file)[0]
    output_file_ts = f"{base_out}_{timestamp}_{MODEL_NAME}.json"
    incorrect_file_ts = f"{args.incorrect_file}_{timestamp}_{MODEL_NAME}.json"
    metrics_file_ts = f"{base_out}_{timestamp}_{MODEL_NAME}_metrics.json"

    # Process samples and save results
    results = process_samples(all_samples, llm, output_file_ts)

    # Evaluate results
    metrics = evaluate_results(results)

    # Save evaluation metrics
    with open(metrics_file_ts, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to {metrics_file_ts}")

    # Save incorrect samples
    save_incorrect_samples(results, all_samples, incorrect_file_ts)
    print(f"Incorrect samples saved to {incorrect_file_ts}")

if __name__ == "__main__":
    main()