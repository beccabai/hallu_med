import os
import json
import argparse
import time
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import traceback

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# 模型路径
MODEL_PATH = "/fs-computility/llmit_d/shared/baitianyi/model/Qwen2.5-VL-72B-Instruct"

# 初始化日志目录
os.makedirs("logs", exist_ok=True)

def load_json_data(json_path):
    """加载JSON数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def initialize_model(model_path=MODEL_PATH, tensor_parallel_size=8, max_model_len=32768):
    """初始化模型"""
    print(f"正在初始化模型: {model_path}，使用{tensor_parallel_size}个GPU...")
    
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
        max_tokens=256,
    )
    
    return llm, processor, sampling_params

def analyze_caption_differences(complete_caption, new_caption, llm, processor, sampling_params, max_retries=3):
    """分析两个描述之间的差异并进行分类"""
    
    # 构建提示
    prompt_template = """Given these two image captions and a specific type of difference:

Caption 1: {complete_caption}
Caption 2: {new_caption}

Type class:
- Object: Adding or removing entities (people, animals, food, vehicles, objects)
- Attribute: Changing visual properties (color, material, size, shape)
- Scene: Modifying backgrounds or settings
- Spatial Relation: Altering physical arrangements (right, left, on top, below, inside)
- Action Relation: Changing interactions between entities
- Part Relation: Modifying part-whole relationships (body parts, clothing, accessories)
- Counting: Adjusting quantity of entities
- Differentiation: Distinguishing objects by attributes
- Comparison: Changing relative characteristics
- Negation: Removing elements
- Universality: Applying changes to all members of a group

Identify only the most obvious differences between these captions and choose a Type class the difference belong to.

Respond with just the key differences, using one sentence and one type class
Please strctly follow this format:"TYPE:XXX\nDIFFERENCE:XXXX"
"""

    prompt = prompt_template.format(
        complete_caption=complete_caption,
        new_caption=new_caption
    )
    
    # 准备消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # 使用模型的聊天模板处理消息
    formatted_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # 带有重试机制的生成
    for retry in range(max_retries):
        try:
            outputs = llm.generate([formatted_prompt], sampling_params=sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            
            # 尝试提取TYPE和DIFFERENCE
            type_match = re.search(r'TYPE:\s*(\w+)', response_text)
            diff_match = re.search(r'DIFFERENCE:\s*(.*)', response_text)
            
            if type_match and diff_match:
                new_type = type_match.group(1).strip()
                difference_description = diff_match.group(1).strip()
                return new_type, difference_description
            else:
                print(f"重试 {retry+1}/{max_retries}: 输出格式不正确: '{response_text}'")
                time.sleep(1)
        except Exception as e:
            print(f"重试 {retry+1}/{max_retries} 出错: {str(e)}")
            time.sleep(2)
    
    # 如果所有重试都失败，返回错误
    return "Error", "无法分析描述差异"

def judge_edit_match(edit_prompt, difference_description, llm, processor, sampling_params, max_retries=3):
    """判断差异描述是否与编辑提示匹配"""
    
    # 构建提示
    prompt_template = """Determine if the described difference matches the edit prompt:

Edit Prompt: {edit_prompt}
Described Difference: {difference_description}

Assess whether the described difference correctly reflects what was requested in the edit prompt. 

Answer with 1. Only "Yes" if they match. No explanation needed.
            2. Only "No" if there's no obvious change. No explanation needed.
"""

    prompt = prompt_template.format(
        edit_prompt=edit_prompt,
        difference_description=difference_description
    )
    
    # 准备消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # 使用模型的聊天模板处理消息
    formatted_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # 带有重试机制的生成
    for retry in range(max_retries):
        try:
            outputs = llm.generate([formatted_prompt], sampling_params=sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            
            # 提取Yes或No
            if "yes" in response_text.lower():
                return "Yes"
            elif "no" in response_text.lower():
                return "No"
            else:
                print(f"重试 {retry+1}/{max_retries}: 输出格式不正确: '{response_text}'")
                time.sleep(1)
        except Exception as e:
            print(f"重试 {retry+1}/{max_retries} 出错: {str(e)}")
            time.sleep(2)
    
    # 如果所有重试都失败，返回错误
    return "Error: 无法判断差异是否匹配编辑提示"

def log_error(error_msg, item_id, error_file="./logs/caption_edit_judge_error.txt"):
    """记录错误信息到文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_file, 'a') as f:
        f.write(f"[{timestamp}] 处理 {item_id} 时出错\n")
        f.write(f"错误信息: {error_msg}\n")
        f.write("-" * 80 + "\n")
    print(f"错误已记录到 {error_file}")

def save_checkpoint(results, processed_count, output_path, shard_id=None):
    """保存检查点"""
    suffix = f"_shard{shard_id}" if shard_id is not None else ""
    checkpoint_path = f"{os.path.splitext(output_path)[0]}{suffix}_checkpoint.json"
    
    checkpoint_data = {
        "processed_count": processed_count,
        "results": results
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=4)
    
    print(f"检查点已保存: {checkpoint_path} (已处理: {processed_count} 项)")
    return checkpoint_path

def load_checkpoint(output_path, shard_id=None):
    """加载检查点（如果存在）"""
    suffix = f"_shard{shard_id}" if shard_id is not None else ""
    checkpoint_path = f"{os.path.splitext(output_path)[0]}{suffix}_checkpoint.json"
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data["results"], checkpoint_data["processed_count"]
        except Exception as e:
            print(f"加载检查点时出错: {str(e)}")
    
    return [], 0

def process_data_shard(data, output_path, shard_id=None, total_shards=None, 
                      checkpoint_interval=10, tensor_parallel_size=8):
    """处理数据分片"""
    # 如果指定了分片，则只处理该分片的数据
    if shard_id is not None and total_shards is not None:
        shard_size = len(data) // total_shards
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size if shard_id < total_shards - 1 else len(data)
        data_shard = data[start_idx:end_idx]
        print(f"处理分片 {shard_id+1}/{total_shards}，共 {len(data_shard)} 项 (总数据 {len(data)} 项)")
    else:
        data_shard = data
        print(f"处理完整数据集，共 {len(data_shard)} 项")
    
    # 加载检查点
    results, start_idx = load_checkpoint(output_path, shard_id)
    
    if start_idx > 0:
        print(f"从检查点恢复: 已处理 {start_idx} 项")
    
    # 初始化模型
    llm, processor, sampling_params = initialize_model(tensor_parallel_size=tensor_parallel_size)
    
    # 处理每个条目
    for idx, item in enumerate(tqdm(data_shard[start_idx:], initial=start_idx, total=len(data_shard)), start=start_idx):
        # 创建新的结果项目，保留原始数据
        result_item = {k: v for k, v in item.items()}
        
        try:
            # 1. 分析描述差异
            complete_caption = item["complete_caption"]
            new_caption = item["new_caption"]
            
            if "CAPTION: " in new_caption:
                new_caption = new_caption.replace("CAPTION: ", "")
            
            new_type, difference_description = analyze_caption_differences(
                complete_caption,
                new_caption,
                llm,
                processor,
                sampling_params
            )
            
            # 添加差异分析结果到输出项
            result_item["difference_description"] = difference_description
            result_item["new_type"] = new_type
            
            # 2. 判断差异是否与编辑提示匹配
            edit_prompt = item["edit_prompt"]
            
            edit_judge = judge_edit_match(
                edit_prompt,
                difference_description,
                llm,
                processor,
                sampling_params
            )
            
            # 添加编辑判断结果到输出项
            result_item["edit_judge"] = edit_judge
            
        except Exception as e:
            error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
            log_error(error_msg, idx)
            
            # 添加错误信息
            result_item["difference_description"] = "Error: " + str(e)
            result_item["new_type"] = "Error"
            result_item["edit_judge"] = "Error"
        
        # 添加到总结果列表
        results.append(result_item)
        
        # 定期保存检查点
        if (idx + 1) % checkpoint_interval == 0:
            save_checkpoint(results, idx + 1, output_path, shard_id)
    
    # 保存最终结果
    suffix = f"_shard{shard_id}" if shard_id is not None else ""
    final_output_path = f"{os.path.splitext(output_path)[0]}{suffix}.json"
    
    final_data = {
        "processed_count": len(results),
        "results": results
    }
    
    with open(final_output_path, 'w') as f:
        json.dump(final_data, f, indent=4)
    
    # 同时保存最终检查点
    save_checkpoint(results, len(data_shard), output_path, shard_id)
    
    print(f"已完成分片 {shard_id if shard_id is not None else '全部'} 的处理")
    print(f"总结果保存到: {final_output_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析描述差异并判断编辑匹配度")
    parser.add_argument("--input_json", type=str, default='/fs-computility/llmit_d/shared/baitianyi/tldr/Data/visualgenome/filter_edit_resultsv2_batch_3/cover_two_caption.json', help="输入的JSON数据路径")
    parser.add_argument("--output_path", type=str, default='/fs-computility/llmit_d/shared/baitianyi/tldr/Data/visualgenome/filter_edit_resultsv2_batch_3/caption_w_edit_judge.json', help="输出路径")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="检查点间隔（每处理N项保存一次）")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="GPU数量")
    parser.add_argument("--shard_id", type=int, default=0, help="要处理的分片ID（从0开始）")
    parser.add_argument("--total_shards", type=int, default=1, help="总分片数")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    
    # 加载数据
    data = load_json_data(args.input_json)
    if "results" in data:
        data = data["results"]
    
    # 处理指定的分片或全部数据
    process_data_shard(
        data,
        args.output_path,
        args.shard_id,
        args.total_shards,
        args.checkpoint_interval,
        args.tensor_parallel_size
    )