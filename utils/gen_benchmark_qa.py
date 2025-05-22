import json
import random
import requests
import os

# Question variations for image comparison
variations = [
    "What is the primary difference between the two images?",
    "What is the main distinction between these two images?",
    "What's the key difference between these two pictures?",
    "Can you identify the principal disparity between these two images?",
    "How do these two images primarily differ from each other?",
    "What's the most significant contrast between these two photos?",
    "What distinguishes one image from the other?",
    "What is the fundamental difference you can see between these two visuals?",
    "What's the most notable variation between these two pictures?"
]

def call_model_api(prompt, difference_description):
    """Call API to generate correct answers"""
    api_key = ""  # Replace with your API key
    base_url = ""
    model = "gpt-4o"
    
    url = f"{base_url}/v1/chat/completions"
    full_prompt = f"{prompt}\n\nInput difference description: \"{difference_description}\""
    
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": 300
    })
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if "ANSWER:" in content:
            answer = content.split("ANSWER:")[1].strip()
            return answer
        else:
            return f"In the second image, {difference_description} compared to the first image."
            
    except Exception as e:
        print(f"API call error: {str(e)}")
        return f"In the second image, {difference_description} compared to the first image."

def create_answer_generation_prompt():
    """Create prompt for answer generation"""
    prompt = """
You are analyzing the differences between two images described by captions. Your task is to generate a clear answer that describes how the second image differs from the first image.

Input: A difference description that compares elements between two images.
Output: A precise statement that explains how the second image (Caption 2) differs from the first image (Caption 1).

Your output MUST follow this exact format:
ANSWER: [Your description of how the second image differs from the first image]

Example:
Input difference description: "The deer on the far right is grazing with his head facing toward the grass in the first caption, while in the second caption, it stands alone facing slightly to the right."
Output: 
ANSWER: In the second image, the deer on the far right is standing alone facing slightly to the right, whereas in the first image, the deer was grazing with its head facing toward the grass.

Important requirements:
1. Always frame your answer to describe how the second image differs from the first image
2. Use precise language to describe the specific differences 
3. Begin with "In the second image" to maintain clarity
4. Always follow the ANSWER: format exactly as shown
"""
    return prompt

def process_json_data(json_path):
    """Process JSON data and add questions with answers"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    answer_prompt = create_answer_generation_prompt()
    
    for i, sample in enumerate(data['results']):
        sample['question'] = random.choice(variations)
        
        difference_desc = sample.get('difference_description', '')
        if difference_desc:
            print(f"Processing sample {i+1}/{len(data['results'])}: Generating correct answer...")
            sample['right_answer'] = call_model_api(answer_prompt, difference_desc)
            print(f"Sample {i+1} completed")
        else:
            sample['right_answer'] = "No difference description available."
    
    return data

def save_processed_data(data, output_path):
    """Save processed data to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Processed data saved to {output_path}")

def main():
    input_json_path = ""
    output_json_path = ""
    processed_data = process_json_data(input_json_path)
    save_processed_data(processed_data, output_json_path)

if __name__ == "__main__":
    main()