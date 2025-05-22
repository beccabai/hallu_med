# Codes for 'Hallucination at a Glance: Controlled Visual Edits and Fine-Grained Multimodal Learning'
The edited images are available in our anonymous [Google Drive link](https://drive.google.com/drive/folders/1iFkiwohCa5kP5mifpRppkW1hqg3nItYD?usp=sharing). You will need to download the original images of Visual Genome and Docci from the web. The rest of the files are in the current code repository.

## Code Organization

```
project/
├── script_gen_edit_prompt.py         # Generate editing prompts
├── script_gen_vg_description.py      # Generate full VG descriptions from multiple sentences
├── script_filter_images.py           # Filter images suitable for editing
├── script_get_complete_caption.py    # Generate complete original captions to capture all objects in the image
├── script_get_edit_caption.py        # Generate edited captions based on editing prompts
├── script_gen_diff_and_judge.py      # Generate difference descriptions and judge alignment with prompts
├── script_gen_sft.py                 # Construct difference descriptions as SFT training data
├── benchmark/
│   ├── MED_bench.json                # Benchmark dataset
│   └── MED_real_set.json             # Real-world evaluation set
```

## Environment Installation
```bash
pip install openai
pip install -q -U google-genai #python >= 3.9
pip install -U transformers
pip install qwen-vl-utils
```

