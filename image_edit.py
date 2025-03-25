import os

from openai import OpenAI
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

from utils import encode_image, get_edit_base_prompt

def gen_edit_prompt(image_path, image_caption=None):
    '''
        使用阿里云百炼平台接口调用qwen2.5-vl来生成编辑图片的prompt
    '''
    client = OpenAI(
        api_key=os.getenv("ALICLOUD_BAILIAN_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    base64_image = encode_image(image_path)
    if image_caption:
        content = get_edit_base_prompt + image_caption
    else:
        content = get_edit_base_prompt
    
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
        
def gen_edit_image(prompt):
    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_KEY"))

    contents = prompt

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=contents,
        config=types.GenerateContentConfig(
        response_modalities=['image', 'text'],
        responseModalities = ["Text", "Image"]
        )
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            image.save('gemini-native-image.png')
            image.show()

if __name__ == "__main__":
    #TODO: 根据数据集接口批量获取图片
    image_path = '../dog_biscuit.png'
    
    prompt = gen_edit_prompt(image_path)
    
    print(prompt)
    
    # gen_edit_image(prompt)