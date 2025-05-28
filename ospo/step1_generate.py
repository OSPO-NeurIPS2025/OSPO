""" README: Step1. Prompt Generation (batch size 1)"""

import os
import re
import yaml
import json
import torch
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, seed_everything
from peft import LoraConfig, get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.prompt.template_element import get_prompt_element
from ospo.base import get_model, get_eval_trainer
from ospo.utils import build_config

class JanusProTestWrapper(LightningModule):
    def __init__(self, args, model, tokenizer, processor, category, max_len):
        super().__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor  
        self.category = category
        self.max_len = max_len

        self.object_prompt = get_prompt_element(category, self.processor)
        self.element_set = set()

    def setup(self, stage=None):
        pass

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        
        if len(self.element_set) >= self.max_len:
            return 
        
        answer = self.generate(prompt=self.object_prompt)

        if self.category not in ["non-spatial", "complex"]:
            stop_words = ['and', 'or', '/', '-', 'color', 'shape', 'texture', 'spatial']
            try:
                answer = answer.split(',')
                answer = [a.strip().lower() for a in answer if a.strip() and re.fullmatch(r'[a-zA-Z ]+', a.strip())]
                filtered_answer = [word for word in answer if all(stop not in word for stop in stop_words)]
                self.element_set.update(filtered_answer)
            except Exception as e:
                print(f"Failed to process answer: {answer}\nError: {e}")

        else:
            try:
                answer = answer.strip().lower()
                self.element_set.add(answer)
            except Exception as e:
                print(f"Failed to process answer: {answer}\nError: {e}")

        
    def on_test_epoch_end(self):
        save_path = self.config.save_path
        with open(os.path.join(save_path, f'{self.category}_element.json'), 'w') as f:
            json.dump(list(self.element_set), f, indent=4)
        print(f'# Generated [{self.category}] elements: {len(self.element_set)}')


    @torch.inference_mode()
    def generate(self, prompt):
        generation_config = {
            'do_sample': True,
            'temperature': 1.3,
            'max_new_tokens': 256
        }

        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(self.device)
        
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids).unsqueeze(0)

        outputs = self.model.language_model.generate(inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                **generation_config
                )
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer


def get_dataloader(category, max_len=70):

    if category == "object":
        max_len = 120
    elif category == "spatial":
        max_len = 40
    elif category == "non-spatial" or category == "complex":
        max_len = 4000

    # dataset length = max_len (but, dummy data)
    dataset = list(range(max_len))  
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=lambda batch: batch,
                            shuffle=False)
    return max_len, dataloader



def main(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    seed_everything(config.seed, workers=True)

    if config.save_path is not None:
        os.makedirs(config.save_path, exist_ok=True)

    vl_chat_processor, tokenizer, model = get_model(model_path=config.model_path, cache_dir=config.cache_dir)
    max_len, dataloader = get_dataloader(category=config.category)

    if config.ckpt_path is not None:
        print("# Load model with checkpoint.")
        ckpt_dir = os.path.dirname(config.ckpt_path)
        ckpt_config_path = os.path.join(ckpt_dir, "config.yaml")
        with open(ckpt_config_path, "r") as file:
            ckpt_config = yaml.safe_load(file)

        # Extract LoRA config
        lora_config = LoraConfig(
            r=ckpt_config["lora"].get("lora_rank"),
            lora_alpha=ckpt_config["lora"]["lora_alpha"],
            target_modules=ckpt_config["lora"]["target_modules"],
            lora_dropout=ckpt_config["lora"]["lora_dropout"],
            modules_to_save=ckpt_config["lora"].get("modules_to_save")                     
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        model = JanusProTestWrapper.load_from_checkpoint(checkpoint_path=config.ckpt_path, 
                                                        args=args,
                                                        model=model,
                                                        tokenizer=tokenizer,
                                                        processor=vl_chat_processor,
                                                        category=config.category, 
                                                        max_len=max_len,
                                                        strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    else:
        print("# Load base model.")
        model = JanusProTestWrapper(args=args,
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    processor=vl_chat_processor,
                                    category=config.category,
                                    max_len=max_len)


    trainer = get_eval_trainer(args, device)
    trainer.test(model, dataloaders=dataloader)



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
