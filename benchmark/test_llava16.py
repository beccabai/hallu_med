import json
import os
import argparse
import re
from pathlib import Path
import torch
import random
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from datetime import datetime
from collections import defaultdict

def initialize_model(model_path, lora_path=None):
    """Initialize LLaVA-Next model using transformers"""
    print(f"Initializing model: {model_path}")
    
    # Load model with appropriate configuration
    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Load LoRA weights if provided
    if lora_path:
        print(f"Loading LoRA from: {lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
    
    # Initialize processor
    processor = LlavaNextProcessor.from_pretrained(model_path)
    
    print("Model initialization complete")
    return model, processor

def ask_question_with_images(model, processor, image_paths, question, options=None):
    """Use LLaVA-Next model to answer questions about two images"""
    
    # Load images
    images = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path)
            images.append(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return "Error: Could not load image"
    
    # Build prompt
    content = "Please carefully observe these two images and answer the following question:\n\n" + question
    
    # Add options to the prompt if available
    if options:
        content += "\n\nOptions:\n"
        for option in options:
            content += f"{option}\n"
        content += "\nPlease answer strictly in the following format: start with 'Answer:', then give the letter of the correct option (without extra punctuation or spaces), For example, 'Answer: A\n'."
    
    # Construct the conversation for LLaVA-Next format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": content}
            ],
        }
    ]
    
    # Add images to the conversation
    for image in images:
        conversation[0]["content"].append({"type": "image"})
    
    # Apply chat template to get properly formatted prompt
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)
    
    # Generation
    generation_args = {
        "max_new_tokens": 20,
        "temperature": 0.8,
        "top_p": 0.4,
        "repetition_penalty": 1.05
    }
    
    with torch.no_grad():
        output = model.generate(**inputs, **generation_args)
    
    # Decode the response
    response_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (remove the initial prompt)
    assistant_response = response_text.split("ASSISTANT:")[-1].strip()
    
    return assistant_response

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
    if not response:
        return _random_choice_from_options(options)
        
    # Check for the specific pattern '. [/INST] Answer: X'
    inst_pattern = re.search(r"\[\s*/INST\s*\]\s*Answer:\s*([A-Za-z])", response)
    if inst_pattern:
        return inst_pattern.group(1).upper()
    
    # Look for the explicit format "Answer: X"
    answer_pattern = re.search(r"Answer:\s*([A-Za-z])", response)
    if answer_pattern:
        return answer_pattern.group(1).upper()
    
    # Check if any line contains "Answer:" followed by a letter
    lines = response.strip().split('\n')
    for line in lines:
        if "Answer:" in line:
            # Find the first letter after "Answer:"
            answer_part = line.split("Answer:")[1].strip()
            if answer_part and answer_part[0].isalpha():
                return answer_part[0].upper()
    
    # If the entire response is just a single letter
    if response.strip().isalpha() and len(response.strip()) == 1:
        return response.strip().upper()
        
    # If no match found, return random choice
    return _random_choice_from_options(options)

def _random_choice_from_options(options=None):
    """Based on available options, choose a random answer letter"""
    # If options provided, extract letters from them
    if options:
        possible_answers = [option[0].upper() for option in options]  # Get first letter of each option
        random_answer = random.choice(possible_answers)
        print(f"Warning: Could not extract answer format. Randomly selecting: {random_answer}")
        return random_answer
    
    # If no options provided, randomly select A-D
    random_answer = random.choice(['A', 'B', 'C', 'D'])
    print(f"Warning: Could not extract answer format. Randomly selecting: {random_answer}")
    return random_answer

def process_samples(samples, model, processor, output_file=None):
    """Process all samples and return results"""
    results = []
    
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}")
        
        question = sample["question"]
        options = sample["options"]
        answer = sample.get("answer", None)  # Get correct answer (if available)
        question_type = sample.get("type", "unknown")  # Get question type
        
        # Extract image paths from JSON format
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
        response = ask_question_with_images(model, processor, valid_paths, question, options)
        
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
            "type": question_type  # Include question type
        }
        results.append(result)
        
        print(f"Question: {question}")
        print(f"Question Type: {question_type}")
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
    """Evaluate model answer accuracy, including per-type accuracy"""
    correct = 0
    total = 0
    
    # Track accuracy by question type
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    
    for result in results:
        if not result.get("ground_truth"):
            continue
            
        ground_truth = result["ground_truth"].upper()  # Convert to uppercase for comparison
        extracted_answer = result.get("extracted_answer")
        question_type = result.get("type", "unknown")
            
        type_total[question_type] += 1
        total += 1
        
        if extracted_answer == ground_truth:
            correct += 1
            type_correct[question_type] += 1
    
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.2%}")
    
    # Calculate and print per-type accuracy
    type_accuracy = {}
    for qtype in type_total:
        type_acc = type_correct[qtype] / type_total[qtype] if type_total[qtype] > 0 else 0
        type_accuracy[qtype] = type_acc
        print(f"Type '{qtype}' Accuracy: {type_correct[qtype]}/{type_total[qtype]} = {type_acc:.2%}")
    
    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "type_accuracy": {qtype: {"accuracy": acc, "correct": type_correct[qtype], "total": type_total[qtype]} 
                          for qtype, acc in type_accuracy.items()}
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
    parser = argparse.ArgumentParser(description="Use LLaVA-Next model to answer questions about images")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--json_path", type=str, default="./MED_bench.json",
                        help="Path to sample JSON file")
    parser.add_argument("--output_file", type=str, default='./final_output/test.json',
                        help="Path to output results file (will get timestamp appended)")
    parser.add_argument("--incorrect_file", type=str, default='./output/incorrect',
                        help="Path to save incorrect samples (will get timestamp & .json appended)")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of samples to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA weights")
    args = parser.parse_args()
    MODEL_NAME = args.model_path.split("/")[-1]

    # Set random seed
    random.seed(args.seed)

    # Load samples
    all_samples = load_samples(args.json_path, args.num_samples)

    # Derive model name for logging
    model_name = args.model_path.split("/")[-1]
    if args.lora_path:
        model_name += "_lora"
        args.incorrect_file += "_lora"
    print(f"Using model: {model_name}")
    print(f"Ready to process {len(all_samples)} samples")

    # Initialize model and processor
    model, processor = initialize_model(
        model_path=args.model_path,
        lora_path=args.lora_path
    )

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build timestamped filenames
    base_out      = os.path.splitext(args.output_file)[0]
    output_file_ts    = f"{base_out}_{timestamp}_{MODEL_NAME}.json"
    incorrect_file_ts = f"{args.incorrect_file}_{timestamp}_{MODEL_NAME}.json"
    metrics_file_ts   = f"{base_out}_{timestamp}_{MODEL_NAME}_metrics.json"

    # Process samples and save results
    results = process_samples(all_samples, model, processor, output_file_ts)

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