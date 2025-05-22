import json

def filter_high_similarity(json_file_path, threshold=0.95):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    filtered_results = [item for item in data['results'] if item.get('CLIP_Similarity', 0) > threshold]
    
    filtered_data = {
        "processed_count": data['processed_count'],
        "results": filtered_results,
        "filtered_count": len(filtered_results)
    }
    
    return filtered_data

# 使用示例
if __name__ == "__main__":
    file_path = "/path/to/json"
    filtered_data = filter_high_similarity(file_path)
    
    print(f"{filtered_data['processed_count']}")
    print(f"{filtered_data['filtered_count']}")
    
    with open("./new_benchmark/benchmark_candidate_3.json", 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)