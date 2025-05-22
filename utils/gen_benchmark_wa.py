import os
import json
import re
import random
import requests
from tqdm import tqdm

def call_gpt_for_wrong_answers(difference_description, right_answer):
    prompt = f"""Please create exactly 3 completely incorrect statements about the difference between two images that are similar to the correct answer but wrong in subtle ways.

Context:
- The first image (Caption 1) and second image (Caption 2) show the same scene with a specific difference
- The actual difference is: "{difference_description}"
- The correct statement that describes this difference is: "{right_answer}"

Your task:
1. Create exactly 3 statements that are deceptively similar to the correct answer but entirely incorrect
2. Each wrong answer should be closely related to the actual difference, but contain crucial errors
3. Make the wrong answers similar enough to the right answer that they could confuse someone
4. The statements should sound plausible and require careful attention to distinguish from the correct answer
5. Each wrong answer should suggest a specific difference between "the first image" and "the second image"
6. Use similar wording, structure, and level of detail as the correct answer

Output format:
```
wrong_answer1: [Your first wrong answer here]
wrong_answer2: [Your second wrong answer here]
wrong_answer3: [Your third wrong answer here]
```

Remember, your statements should be subtly wrong in ways that make them challenging to distinguish from the correct answer!"""

    # Call API
    api_key = ""
    base_url = ""
    model = "gpt-4o"
    
    url = f"{base_url}/v1/chat/completions"
    
    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500
    })
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response_json = response.json()
        result = response_json["choices"][0]["message"]["content"]
        
        # Extract the three wrong answers using regex
        wrong_answers = []
        
        pattern = r"wrong_answer\d+:\s*(.*?)$"
        matches = re.findall(pattern, result, re.MULTILINE)
        
        if len(matches) >= 3:
            wrong_answers = matches[:3]
        else:
            # Fallback extraction method
            lines = result.strip().split('\n')
            for line in lines:
                if ':' in line and not line.startswith('```'):
                    parts = line.split(':', 1)
                    if len(parts) == 2 and 'wrong_answer' in parts[0]:
                        wrong_answers.append(parts[1].strip())
        
        # If we still don't have 3 answers, generate some generic ones
        while len(wrong_answers) < 3:
            wrong_answers.append(f"In the second image, there is a completely different scene compared to the first image.")
            
        return wrong_answers[:3]  # Ensure we return exactly 3 answers
    
    except Exception as e:
        print(f"Error calling GPT API: {str(e)}")
        # Fallback wrong answers
        return [
            "In the second image, the colors are completely inverted compared to the first image.",
            "In the second image, there are additional people present that weren't in the first image.",
            "In the second image, the weather conditions changed from sunny to rainy compared to the first image."
        ]

def process_json_data(json_file_path, output_file_path):
    """Process the JSON data, adding wrong answers and options"""
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data['results'])} samples...")
    
    for item in tqdm(data['results']):
        if 'right_answer' in item and 'difference_description' in item:
            # Get wrong answers using GPT
            wrong_answers = call_gpt_for_wrong_answers(
                item['difference_description'], 
                item['right_answer']
            )
            
            # Add wrong_answers field
            item['wrong_answers'] = wrong_answers
            
            # Create options array with randomized positions
            options = [
                f"A. {wrong_answers[0]}",
                f"B. {item['right_answer']}",
                f"C. {wrong_answers[1]}",
                f"D. {wrong_answers[2]}"
            ]
            
            # Shuffle options while tracking the correct answer position
            correct_option_index = 1  # B is initially the correct answer
            
            # Create a list of indices to shuffle
            indices = list(range(len(options)))
            random.shuffle(indices)
            
            # Reorder options based on shuffled indices
            new_options = [options[i] for i in indices]
            
            # Find the new index of the correct answer
            new_correct_index = indices.index(correct_option_index)
            right_option = chr(65 + new_correct_index)  # Convert to letter (A, B, C, D)
            
            # Add options and right_option fields
            item['options'] = new_options
            item['right_option'] = right_option
    
    # Save the updated data
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete. Output saved to {output_file_path}")

# Main execution
if __name__ == "__main__":
    input_file = ""  # Change to your input file path
    output_file = ""  # Change to your desired output file
    
    process_json_data(input_file, output_file)