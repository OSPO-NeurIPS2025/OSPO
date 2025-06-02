import os
import json
import random
import argparse
import inflect

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.wrapper import JanusProElementGenWrapper
from ospo.datamodule import GenerationDataModule
from ospo.utils.generate import get_trainer
from ospo.utils.model import get_model, get_lora_config
from ospo.utils.common import read_json, save_json, build_config


def load_element(base_path):
    object_element = read_json(os.path.join(base_path, 'object_element.json'))
    color_element = read_json(os.path.join(base_path, 'color_element.json'))
    shape_element = read_json(os.path.join(base_path, 'shape_element.json'))
    texture_element = read_json(os.path.join(base_path, 'texture_element.json'))
    spatial_element = read_json(os.path.join(base_path, 'spatial_element.json'))
    
    return object_element, color_element, shape_element, texture_element, spatial_element


def load_prompt(base_path):
    non_spatial_prompt = read_json(os.path.join(base_path, 'non-spatial_element.json'))
    complex_prompt = read_json(os.path.join(base_path, 'complex_element.json'))

    return non_spatial_prompt, complex_prompt


def construct_prompt(object_element_list, binding_element_list, generate_type, generate_num=1000):
    prompt_set = set()
    p = inflect.engine()

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


def combine_metadata(category, prompt_list): 
    # itme_id, category, sub_category, prompt
    category2idx = {"attribute": 0, "layout": 1, "non-spatial": 2, "complex": 3}

    for i, sample in enumerate(prompt_list):
        item_id = f"{category2idx[category]}{i:06d}"
        sample["item_id"] = item_id

    return prompt_list


def combine_prompt(config):
    base_prompt_list = {
        "attribute": [],
        "layout": [],
        "non-spatial": [],
        "complex": []
        }
    category_num_dict = config.category_num

    # Load element/prompt data (generated in Step1)
    object_element, color_element, shape_element, texture_element, spatial_element = load_element(config.save_path)
    non_spatial_prompt, complex_prompt = load_prompt(config.save_path)
    attributes = [color_element, shape_element, texture_element]

    for sub_category, num in category_num_dict.items():
        if sub_category == "attribute1_color":
            category = "attribute"
            prompt_list = construct_prompt(object_element, color_element, "attribute1", num)
        elif sub_category == "attribute1_shape":
            category = "attribute"
            prompt_list = construct_prompt(object_element, shape_element, "attribute1", num)
        elif sub_category == "attribute1_texture":
            category = "attribute"
            prompt_list = construct_prompt(object_element, texture_element, "attribute1", num)
        elif sub_category == "attribute2":
            category = "attribute"
            prompt_list = construct_prompt(object_element, attributes, "attribute2", num)
        elif sub_category == "layout1":
            category = "layout"
            prompt_list = construct_prompt(object_element, spatial_element, "layout1", num)
        elif sub_category == "layout2":
            category = "layout"
            prompt_list = construct_prompt(object_element, [], "layout2", num)
        elif sub_category == "layout3":
            category = "layout"
            prompt_list = construct_prompt(object_element, [], "layout3", num)
        elif sub_category == "non-spatial":
            category = sub_category
            prompt_list = non_spatial_prompt
            assert len(prompt_list) == num, f"Expected {num} prompts for non-spatial, but got {len(prompt_list)}."
        elif sub_category == "complex":
            category = sub_category
            prompt_list = complex_prompt
            assert len(prompt_list) == num, f"Expected {num} prompts for complex, but got {len(prompt_list)}."

        for prompt in prompt_list:
            prompt = prompt.strip()
            base_prompt_list[category].append({
                "prompt": prompt,
                "category": category,
                "sub_category": sub_category
            })

    output = []
    for c, v in base_prompt_list.items():
        output.extend(combine_metadata(c, v))
    print(f"\n*** Total number of base prompt (including non-spatial, complex): {len(output)} ***")

    # save
    save_json(config.save_path, 'base_prompt', output)


def get_dataloader(config):
    datamodule = GenerationDataModule(config, step=1)  # step=1 for element/base prompt generation
    dataloader = datamodule.gen_dataloader()
    return dataloader 


def main(config):

    if config.batch_size > 1 or config.world_size > 1:
        raise NotImplementedError("Batch size > 1 and World size > 1 are not supported in this step.")

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    seed_everything(config.seed, workers=True)
    if config.save_path is not None:
        os.makedirs(config.save_path, exist_ok=True)

    dataloader = get_dataloader(config)
    vl_chat_processor, tokenizer, model = get_model(mode='generate', config=config)

    if config.ckpt_path is not None:
        print("# Load model with checkpoint.")
        lora_config = get_lora_config(config.ckpt_path)
        
        model.language_model = get_peft_model(model.language_model, lora_config)
        model = JanusProElementGenWrapper.load_from_checkpoint(checkpoint_path=config.ckpt_path, 
                                                        config=config,
                                                        model=model,
                                                        tokenizer=tokenizer,
                                                        processor=vl_chat_processor,
                                                        strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    else:
        print("# Load base model.")
        model = JanusProElementGenWrapper(config=config,
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    processor=vl_chat_processor)

    trainer = get_trainer(device, config.world_size)
    trainer.test(model, dataloaders=dataloader)

    # if all elements and prompts are generated, combine them into one.
    generated = ['object_element.json', 'color_element.json', 'shape_element.json', 'texture_element.json',  'spatial_element.json', 'non-spatial_element.json', 'complex_element.json']
    if all(os.path.exists(os.path.join(config.save_path, f)) for f in generated):
        print("All elements and prompts are generated. Start combining all.")
        combine_prompt(config)
    print("(Step 1) Base prompt generation completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='configs/step1.yaml')
    args, unknown = parser.parse_known_args()  
    
    config = build_config(cfg_path=args.cfg_path)
    if config.category is None:
        raise ValueError("Please specify the category in the config file.")
    else:
        print("# Category:", config.category)

    main(config)
