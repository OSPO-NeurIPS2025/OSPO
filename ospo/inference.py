import os
import sys
import numpy as np
import PIL.Image
from tqdm import tqdm
import random
import time
import json
import yaml
import argparse
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelSummary

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf 

from janus.models import MultiModalityCausalLM, VLChatProcessor

import traceback

class Inference(Dataset):
    def __init__(self, config):
        self.config = config

        with open(config.data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class JanusProTestWrapper(LightningModule):
    def __init__(self, config, vl_chat_processor, model):
        super().__init__()

        self.config = config
        self.vl_chat_processor = vl_chat_processor
        self.model = model

        self.error_data = []

        self.seed_list = config.seed
        self.parallel_size = config.model.generation_cfg.parallel_size
        self.temperature = config.model.generation_cfg.temperature
        self.cfg_weight = config.model.generation_cfg.cfg_weight


    def setup(self, stage=None):
        pass

    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        
        prompt_list = []
        final_path_list = []

        for sample in batch:
            prompt = self.get_prompt(sample)
        
            save_path = os.path.join(self.config.save_path)
            os.makedirs(save_path, exist_ok=True)
            fname = f"{sample}.png"

            final_path = os.path.join(save_path, fname)
        
            if os.path.exists(final_path):
                continue
            else:
                prompt_list.append(prompt)
                final_path_list.append(final_path)

        if len(final_path_list) == 0:
            return 

        try:
            for seed in self.seed_list:
                self.generate_batch(
                    prompt_list=prompt_list,
                    save_path_list=final_path_list,
                    seed=seed
                )
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.error_data.append(sample)


    @torch.inference_mode()
    def generate_batch(
        self,
        prompt_list: List,
        save_path_list: List,
        seed: int = None,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        batch_size = len(prompt_list) 
        input_ids_list = []
        max_len = 0 # for padding

        for prompt in prompt_list:
            input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)

            max_len = max(max_len, len(input_ids))
            input_ids_list.append(input_ids)

        tokens = torch.zeros((batch_size*2, max_len), dtype=torch.int).to(self.device)
        attention_masks = torch.ones((batch_size*2, max_len), dtype=torch.long).to(self.device)
    
        for i in range(batch_size*2):
            pad_len = max_len - len(input_ids_list[i//2])
            tokens[i, pad_len:] = input_ids_list[i//2]
            tokens[i, :pad_len] = self.vl_chat_processor.pad_id
            attention_masks[i, :pad_len] = 0
            if i % 2 != 0:
                tokens[i, pad_len+1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)


        # Generate

        generated_tokens = torch.zeros((batch_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(inputs_embeds=inputs_embeds, 
                                                    attention_mask=attention_masks, 
                                                    use_cache=True, 
                                                    past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + self.cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / self.temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

            # cat attention mask
            new_mask = torch.ones((attention_masks.shape[0], 1), dtype=attention_masks.dtype, device=attention_masks.device)  
            attention_masks = torch.cat([attention_masks, new_mask], dim=1)

        dec = self.model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[batch_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        
        visual_img = np.zeros((batch_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        # Save img
        if self.parallel_size == 1:
            for inner_idx, image in enumerate(visual_img):
                try:
                    PIL.Image.fromarray(image).save(save_path_list[inner_idx].replace(".png",f"_{seed}.png"))
                    
                except OSError:
                    idx_in_path = save_path_list[inner_idx].split("_")[1] # 01.png
                    alternative_path = f"longprompt_{idx_in_path}"

                    PIL.Image.fromarray(image).save(alternative_path)

        else:
            raise NotImplementedError("parallel_size > 1, not supported.") # we only use parallel_size = 1
            
    def on_test_epoch_end(self):
        print(f"Error case: {len(self.error_data)}")

        save_path = os.path.join(self.config.save_path, "error_sample.json")
        with open(save_path, "w") as f:
            json.dump(save_path, f, indent=4)    

    def get_prompt(self, text):

        conversation = [
            {
                "role": "User",
                "content": text,
            },
            {"role": "Assistant", "content": ""},
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag
        return prompt


def get_model(config):

    model_path = config.model_path
    dtype = torch.bfloat16 if config.precision == 'bf16' else torch.float32
    cache_dir = config.cache_dir

    if model_path == None:
        print(f"Model is loaded from cache_dir: {cache_dir}")
    else:
        print(f"Model is loaded from model_path: {model_path}")

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).eval()
    return vl_chat_processor, tokenizer, vl_gpt


def get_dataloader(config):
    
    dataset=Inference(config)

    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        collate_fn=lambda batch: batch,
        num_workers=config.num_workers,  
        drop_last=False
    )
    return dataloader


def get_trainer(config, device):
    trainer = Trainer(
        accelerator=device,
        devices=config.world_size,
        strategy="ddp",
        max_epochs=1, # config.experiment.epoch,
        precision=config.precision,
        callbacks=[ModelSummary(max_depth=2)],
        logger=False,
    )
    return trainer

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True) # Janus-8B path
    parser.add_argument("--ckpt_path", type=str, required=True) # ospo ckpt path
    parser.add_argument("--save_path", type=str, default="./results")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=1)

    parser.add_argument("--overrides", type=list, default=[])
    return parser.parse_args()

def load_config():
    
    args = args_parser()

    overrides = [
        f"model.model_path={args.model_path}",
        f"model.ckpt_path={args.ckpt_path}",
        f"save_path={args.save_path}",
        f"trainer.world_size={args.world_size}",
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
    ] + args.overrides
    
    initialize(config_path="../configs", version_base=None)
    config = compose(config_name="inference", overrides=overrides)
    OmegaConf.resolve(config)

    return config


def main():

    config=load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(config.seed[0], workers=True)

    vl_chat_processor, tokenizer, model = get_model(config.model)
    
    config.model.ckpt_path = "/nas2/checkpoints/janus_dpo/final/0507_simpo_base_lr_4e_5_beta_10_gamma_0_5/step=000100.ckpt"
    if not os.path.exists(config.model.ckpt_path):
        raise ValueError("Check model.ckpt_path !")
    
    # Extract LoRA config
    lora_config = LoraConfig(
        r=config.peft.lora_rank,
        lora_alpha=config.peft.lora_alpha,
        target_modules=config.peft.target_modules,
        lora_dropout=config.peft.lora_dropout,
        modules_to_save=config.peft.modules_to_save
    )

    model.language_model = get_peft_model(model.language_model, lora_config)
    model = JanusProTestWrapper.load_from_checkpoint(checkpoint_path=config.model.ckpt_path, 
                                                        config=config, 
                                                        vl_chat_processor=vl_chat_processor, 
                                                        model=model,
                                                        strict=False)

    model.setup("test")
    model.model.language_model = model.model.language_model.merge_and_unload() 

    trainer = get_trainer(config.trainer, device)
    eval_dataloader = get_dataloader(config.data)
    
    start = time.time()
    trainer.test(model, dataloaders=eval_dataloader)
    end = time.time()

    elapsed_time = (end - start) / 60  # Convert seconds to minutes
    print(f"Done ! Time elapsed: {elapsed_time:.2f} minutes")

if __name__ == "__main__":
    main()