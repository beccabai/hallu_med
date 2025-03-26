import os
import json
import argparse
import base64
import math
from tqdm import tqdm
from openai import OpenAI
from utils import encode_image

def get_filter_prompt():
    base_prompt = '''
    Please evaluate the following sample--which includes an image and a caption based on the criteria listed below. For each criterion, respond with 'Yes' if the sample meets the requirement or 'No' if it does not, along with a brief explanation. Finally, provide an overall recommendation on whether this sample is suitable for further processing.
        1.	Image Quality:
        •	Clarity: Is the image sharp, free of blur, and low in noise?
        •	Main Subject: Is there a clearly identifiable primary subject in the image?
        •	Background Complexity: Is the background simple and free of excessive clutter?
        2.	Question-Answer Clarity:
        •	Question Specificity: Is the question clear, specific, and unambiguous?
        •	Answer Uniqueness: Is the answer straightforward and unique, without ambiguity or multiple interpretations?
        3.	Editing Potential:
        •	Identifiable Editable Elements: Does the image contain easily identifiable objects, quantities, or attributes that can be modified (e.g., object replacement, quantity change, attribute alteration)?
        •	Impact on Answer: If an edit were applied, would the original answer clearly become incorrect?
        4.	Interference Factors:
        •	Visual Distractions: Are there any distracting elements or overly complex scenes in the image that might hinder a clear edit?
        •	Subjectivity in Text: Does the question or answer include any subjective or open-ended aspects that might interfere with a definitive evaluation?

    Based on your assessment for each criterion, please provide an overall recommendation on whether this sample should be selected for further processing in our pipeline.

    Caption of the image: {caption}
    '''
    
    return base_prompt

def gen_filter_prompt(image_path, image_caption=None):
    '''
        使用阿里云百炼平台接口调用qwen2.5-vl来生成过滤图片的prompt
    '''
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
        model="qwen2.5-vl-32b-instruct",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[{"role": "user","content": [
                {"type": "text","text": content},
                {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}]
        )
    prompt = completion.choices[0].message.content
    
    return prompt

def load_docci_dataset(base_dir="docci", images_dir="docci/images", limit=None):
    """加载DOCCI数据集"""
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
    """检查文本中是否包含'Yes'"""
    return 'Yes' in text

def ensure_dir_exists(file_path):
    """确保文件目录存在"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def split_samples(samples, total_parts, part_index):
    """将样本划分为N份，返回第n份"""
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
    
    ensure_dir_exists(os.path.join(args.output_dir, "temp"))
    
    # 构建输出文件名（包含分片信息）
    filtered_output = os.path.join(args.output_dir, f"filtered_image_docci_part_{args.part_index}.json")
    debug_output = os.path.join(args.output_dir, f"debug_image_docci_part_{args.part_index}.json")
    
    # 1. 读取DOCCI数据集的图片和描述
    print(f"Loading DOCCI dataset from {args.base_dir}...")
    all_samples = load_docci_dataset(base_dir=args.base_dir, images_dir=args.images_dir, limit=args.limit)
    print(f"Loaded {len(all_samples)} total samples")
    
    # 2. 将样本分片并获取当前处理的分片
    try:
        part_samples = split_samples(all_samples, args.total_parts, args.part_index)
        print(f"Processing part {args.part_index} of {args.total_parts} with {len(part_samples)} samples")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    # 创建保存结果的列表
    filtered_results = []  # 通过过滤的样本
    debug_results = []     # 未通过过滤的样本
    
    # 3. 处理每个样本并保存结果
    print("Processing samples...")
    processed_count = 0
    filtered_count = 0
    debug_count = 0
    
    for sample in tqdm(part_samples):
        try:
            processed_count += 1
            
            # 调用模型分析图片和描述
            reason = gen_filter_prompt(sample['image_path'], sample['description'])
            
            # 创建结果对象
            result = {
                "image_name": sample['image_name'],
                "description": sample['description'],
                "reason": reason
            }
            
            # 根据是否包含"Yes"分别保存到不同文件
            if contains_yes(reason):
                filtered_results.append(result)
                filtered_count += 1
                
                # 每10个过滤后的样本保存一次
                if filtered_count % 5 == 0:
                    # 创建临时文件名，避免写入错误导致原文件损坏
                    temp_file = os.path.join(args.output_dir, "temp", f"filtered_temp_{args.part_index}.json")
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(filtered_results, f, ensure_ascii=False, indent=4)
                    # 成功写入后替换原文件
                    os.replace(temp_file, filtered_output)
                    print(f"Checkpoint saved: {filtered_count} filtered results")
            else:
                debug_results.append(result)
                debug_count += 1
                
                # 每5个调试样本保存一次
                if debug_count % 5 == 0:
                    # 创建临时文件名，避免写入错误导致原文件损坏
                    temp_file = os.path.join(args.output_dir, "temp", f"debug_temp_{args.part_index}.json")
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(debug_results, f, ensure_ascii=False, indent=4)
                    # 成功写入后替换原文件
                    os.replace(temp_file, debug_output)
                    print(f"Checkpoint saved: {debug_count} debug results")
            
        except Exception as e:
            print(f"Error processing {sample['image_name']}: {e}")
            with open(os.path.join(args.output_dir, f"errors_part_{args.part_index}.txt"), 'a') as f:
                f.write(f"{sample['image_name']}: {str(e)}\n")
    
    # 保存最终结果
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