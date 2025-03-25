import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_edit_base_prompt():
    base_prompt = """
    You are a professional image prompt engineer specializing in image editing instructions. 

    Analyze the provided image and any accompanying text, then generate a clear editing prompt based on what you see.

    Your editing prompt must follow this exact structure:
    "Hi, This is a picture of [brief description of the current image]. Can you [specific editing instruction] and output the image?"

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
    - "Hi, This is a picture of a dog with biscuits on its head. Can you subtract one biscuit on its head and output the image?"
    - "Hi, This is a picture of a highway. Can you add two vehicles on its left side and output the image?"
    - "Hi, This is a picture of a table. Can you only move the phone out of this scene and output the image?"

    Based solely on what you observe in the provided image, create an appropriate editing prompt now.
    """
    return base_prompt
