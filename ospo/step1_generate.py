""" README: Step1. Prompt Generation (batch size 1)"""

import os
import re
import torch
import argparse
import torch.distributed as dist
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, seed_everything
from peft import get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.prompt.template_element import get_prompt_element
from ospo.base import get_model, get_eval_trainer, get_lora_config
from OSPO.ospo.utils_legacy import build_config, save_json

class JanusProTestWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor, max_len):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor  
        self.category = self.config.category
        self.max_len = max_len

        self.object_prompt = get_prompt_element(self.category, self.processor)
        self.element_set = set()

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
        froot = self.config.save_path
        fname = f'{self.category}_element'

        if self.trainer.world_size == 1:
            save_json(froot, fname, list(self.element_set))
            print(f'# Generated [{self.category}] elements: {len(self.element_set)}')
            return

        # if self.trainer.world_size > 1:
        gathered_element_list = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(gathered_element_list, list(self.element_set))
        self.trainer.strategy.barrier()

        if self.trainer.global_rank == 0:
            merged = set()
            for sublist in gathered_element_list:
                merged.update(sublist)
            save_json(froot, fname, list(merged))
            print(f'# Generated [{self.category}] elements: {len(merged)}')

            
    @torch.inference_mode()
    def generate(self, prompt):
        generation_config = self.config.generation_config
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


def get_dataloader(config):
    if config.max_len is None:
        if config.category == "object":
            max_len = 120
        elif config.category == "spatial":
            max_len = 40
        elif config.category == "non-spatial" or config.category == "complex":
            max_len = 4000
        else:
            max_len = 70
    else:
        max_len = config.max_len

    # dataset length = max_len (but, dummy data)
    dataset = list(range(max_len))  
    dataloader = DataLoader(dataset, 
                            batch_size=config.batch_size, 
                            collate_fn=lambda batch: batch,
                            shuffle=False)
    
    return max_len, dataloader



def main(config):

    if config.batch_size > 1 or config.world_size > 1:
        raise NotImplementedError("Batch size > 1 and World size > 1 are not supported in this step.")

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    seed_everything(config.seed, workers=True)

    if config.save_path is not None:
        os.makedirs(config.save_path, exist_ok=True)

    vl_chat_processor, tokenizer, model = get_model(model_path=config.model_path, cache_dir=config.cache_dir)
    max_len, dataloader = get_dataloader(config)

    if config.ckpt_path is not None:
        print("# Load model with checkpoint.")
        lora_config = get_lora_config(config.ckpt_path)
        
        model.language_model = get_peft_model(model.language_model, lora_config)
        model = JanusProTestWrapper.load_from_checkpoint(checkpoint_path=config.ckpt_path, 
                                                        config=config,
                                                        model=model,
                                                        tokenizer=tokenizer,
                                                        processor=vl_chat_processor,
                                                        max_len=max_len,
                                                        strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    else:
        print("# Load base model.")
        model = JanusProTestWrapper(config=config,
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    processor=vl_chat_processor,
                                    max_len=max_len)


    trainer = get_eval_trainer(device, config.world_size)
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
