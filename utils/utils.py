import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_edit_base_prompt():
    base_prompt = """
    You are a professional image prompt engineer specializing in image editing instructions. 

    Analyze the provided image and any accompanying text, then generate a clear editing prompt based on what you see.

    Description of the image: {}

    Your editing prompt must follow this exact structure:
    "Hi, This is a photo of [brief description of the current image]. Can you [specific editing instruction] and output the image?"

    Choose an appropriate editing instruction from these composition categories:

    BASIC COMPOSITIONS:
    - Object: Adding or removing entities (people, animals, food, vehicles, objects)
    - Attribute: Changing visual properties (color, material, size, shape)
    - Scene: Modifying backgrounds or settings
    - Spatial Relation: Altering physical arrangements (right, left, on top, below, inside)
    - Action Relation: Changing interactions between entities
    - Part Relation: Modifying part-whole relationships (body parts, clothing, accessories)

    ADVANCED COMPOSITIONS:
    - Counting: Adjusting quantity of entities
    - Differentiation: Distinguishing objects by attributes
    - Comparison: Changing relative characteristics
    - Negation: Removing elements
    - Universality: Applying changes to all members of a group

    Examples:
    - "Hi, This is a photo of a dog with biscuits on its head. Can you subtract one biscuit on its head and output the image?"
    - "Hi, This is a picture of a highway. Can you add two vehicles on its left side and output the image?"
    - "Hi, This is a photo of a table. Can you only move the phone out of this scene and output the image?"

    Based solely on what you observe in the provided image, first indicate which composition type you're using (Prioritize COMPOSITIONS other than 'Object'. Only choose 'Object' if other types cannot effectively demonstrate the editing possibilities of the image.), then create the appropriate editing prompt.
    Your response should be in this format:
    ## Type: [composition type]\n## Prompt: [your editing prompt]\n
    """
    return base_prompt

def get_filter_prompt():
    base_prompt = '''
    Please evaluate the following sample--which includes an image and accompanying text (question-answer pair or description)--based on the criteria listed below. Your response must STRICTLY follow the format of the provided example. For each criterion, provide a brief analysis, and conclude with a clear "Yes" or "No" recommendation.

    Caption of the image: {caption}
    ---

    ## **Evaluation**

    **Final Recommendation:** **Yes** or **No**

    **1. Image_quality:**
    - **Clarity:** [Evaluate if the image is sharp, free of blur, and low in noise]
    - **Main_Subject:** [Evaluate if there is a clearly identifiable primary subject in the image]
    - **Background_Complexity:** [Evaluate whether the background is simple and free of excessive clutter]

    **2. Text_Clarity (Description):**
    - **Specifity:** [Evaluate if the description is clear, specific, and unambiguous]
    - **Uniquenes:** [Evaluate if the text provides straightforward, unique information without ambiguity]

    **3. Editing_Potential:**
    - **Identifiable_Editable_Elements:** [Evaluate if the image contains clearly identifiable objects or attributes that can be modified]
    - **Impact_on_Text_Accuracy:** [Evaluate whether applying an edit would make the original text incorrect]

    **4. Interference_Factors:**
    - **Visual_Distractions:** [Evaluate whether there are distracting elements or complex scenes that might hinder a clear edit]
    - **Subjectivity_in_Text:** [Evaluate if the text contains subjective aspects that could interfere with evaluation]

    **Explanation:** [Provide a brief explanation supporting your recommendation]

    ---

    IMPORTANT: Your answer MUST follow this EXACT format, with bold section headers and bullet points as shown above. Your Final Recommendation must clearly state either "**Yes**" or "**No**" in bold.
    '''
    return base_prompt
