""" README: Post-process data from Step1. """

import os
import json
import random
import argparse
import inflect
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils import open_json, save_json, build_config


def load_element(base_path):
    object_element = open_json(os.path.join(base_path, 'object_element.json'))
    color_element = open_json(os.path.join(base_path, 'color_element.json'))
    shape_element = open_json(os.path.join(base_path, 'shape_element.json'))
    texture_element = open_json(os.path.join(base_path, 'texture_element.json'))
    spatial_element = open_json(os.path.join(base_path, 'spatial_element.json'))
    
    return object_element, color_element, shape_element, texture_element, spatial_element


def load_prompt(base_path):
    non_spatial_prompt = open_json(os.path.join(base_path, 'non_spatial_element.json'))
    complex_prompt = open_json(os.path.join(base_path, 'complex_element.json'))

    return non_spatial_prompt, complex_prompt


def construct_prompt(object_element_list, binding_element_list, generate_type, generate_num=1000):
    prompt_set = set()

    prompt_format = {
    "attribute1": "A {} {}",              # A {adj} {noun}
    "attribute2": "A {} {} and a {} {}",  # A {adj} {noun} and a {adj} {noun}
    "layout1": "A {} {} a {}",            # A {noun1} {spatial_relatioin} a {noun2}
    "layout2": "{} {}",                   # {quantity} {object}
    "layout3": "{} {} and {} {}",         # {quantity} {object} and {quantity} {object}
    }

    if generate_type != "layout2":
        while len(prompt_set) < generate_num:
            if generate_type == "attribute1":
                random_object = random.choice(object_element_list)
                selected_binding = random.choice(binding_element_list) # color, shape, texture
                prompt = prompt_format[generate_type].format(selected_binding.strip().lower(), random_object.strip().lower())

            elif generate_type == "attribute2":
                random_object = random.sample(object_element_list, 2)               # no duplicate
                random_binding = random.choices(binding_element_list, k=2)          # allowing duplicate
                while True:
                    selected_binding1 = random.choice(random_binding[0])
                    selected_binding2 = random.choice(random_binding[1])
                    if selected_binding1 != selected_binding2:
                        break
                prompt = prompt_format[generate_type].format(selected_binding1.strip().lower(), random_object[0].strip().lower(),selected_binding2.strip().lower(), random_object[1].strip().lower())

            elif generate_type == "layout1":
                random_object = random.sample(object_element_list, 2)
                selected_binding = random.choice(binding_element_list)
                prompt = prompt_format[generate_type].format(random_object[0].strip().lower(), selected_binding.strip().lower(), random_object[1].strip().lower())
            
            elif generate_type == "layout3":
                random_object = random.sample(object_element_list, 2)
                num1, num2 = random.randint(1,5), random.randint(1,5)

                quantity1 = "A" if num1 == 1 else p.number_to_words(num1).capitalize()
                object1 = random_object[0] if num1 == 1 else p.plural(random_object[0].strip())
                quantity2 = "a" if num2 == 1 else p.number_to_words(num2)
                object2 = random_object[1] if num2 == 1 else p.plural(random_object[1].strip())
        
                prompt = prompt_format[generate_type].format(quantity1, object1, quantity2, object2)

            prompt_set.add(prompt)

        prompt_list = list(prompt_set)

    else: # generate_type == "layout2"
        p = inflect.engine()
        for num in range(1, 30):
            for object in object_element_list:
                object = object.strip()
                if num == 1:
                    prompt = f'A {object}'
                else:
                    prompt = prompt_format[generate_type].format(p.number_to_words(num).capitalize(), p.plural(object.strip()))
                prompt_set.add(prompt)
                if len(prompt_set) == generate_num:
                    break
            if len(prompt_set) == generate_num:
                break

        prompt_list = list(prompt_set)
        random.shuffle(prompt_list)

    print(f"*** Generated [{generate_type}] prompt: {len(prompt_list)} ***")
    return prompt_list


def combine_metadata(sub_category, prompt_list): 
    data_list = [] # itme_id, category, sub_category, prompt

    if "attribute" in sub_category:
        category = "attribute"
    elif "layout" in sub_category:
        category = "layout"
    else:
        category = sub_category
    category2idx = {"attribute": 0, "layout": 1, "non-spatial": 2, "complex": 3}

    for i, sample in enumerate(prompt_list):
        item_id = f"{category2idx[category]}{i:06d}"
        data_list.append({
            "item_id": item_id,
            "prompt": sample,
            "category": category,  
            "sub_category": sub_category
        })
    return data_list


def combine_prompt(config):
    # Combine all prompts across categories (9) with metadata.
    base_prompt_list = []

    # Load element/prompt data (generated in Step1)
    object_element, color_element, shape_element, texture_element, spatial_element = load_element(config.save_path)
    attributes = [color_element, shape_element, texture_element]
    non_spatial_prompt, complex_prompt = load_prompt(config.save_path)

    category_num_dict = {
        'attribute1_color': 666,
        'attribute1_shape': 667,
        'attribute1_texture': 667,
        'attribute2': 2000,
        'layout1': 2000,
        'layout2': 1000,
        'layout3': 1000,
        'non_spatial': 4000,
        'complex': 4000
    }

    for category, num in category_num_dict.items():
        if category == "attribute1_color":
            prompt_list = construct_prompt(object_element, color_element, "attribute1", num)
        elif category == "attribute1_shape":
            prompt_list = construct_prompt(object_element, shape_element, "attribute1", num)
        elif category == "attribute1_texture":
            prompt_list = construct_prompt(object_element, texture_element, "attribute1", num)
        elif category == "attribute2":
            prompt_list = construct_prompt(object_element, attributes, "attribute2", num)
        elif category == "layout1":
            prompt_list = construct_prompt(object_element, spatial_element, "layout1", num)
        elif category == "layout2":
            prompt_list = construct_prompt(object_element, [], "layout2", num)
        elif category == "layout3":
            prompt_list = construct_prompt(object_element, [], "layout3", num)
        elif category == "non_spatial":
            prompt_list = non_spatial_prompt
            assert len(prompt_list) == num, f"Expected {num} prompts for non_spatial, but got {len(prompt_list)}."
        elif category == "complex":
            prompt_list = complex_prompt
            assert len(prompt_list) == num, f"Expected {num} prompts for complex, but got {len(prompt_list)}."

        # generate metadata for each prompt sampe
        data_list = combine_metadata(category, prompt_list)
        base_prompt_list.extend(data_list)

    print(f"*** Total number of base prompt: {len(base_prompt_list)} ***")
    return base_prompt_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='configs/step1.yaml')
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)
    
    base_prompt_list = construct_prompt(config)
    save_json(config.save_path, 'base_prompt_meta_16k.json', base_prompt_list)
