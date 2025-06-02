""" README: Step3. Image Generation """

import os
import argparse
import traceback
from typing import List
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule
from peft import get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.base import get_model, get_lora_config, get_eval_trainer, get_prompt, get_fname
from OSPO.ospo.utils_legacy import save_json, set_seed, build_config
from dataclass.gen_dataset import BaseDataset


class JanusProTestWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # fixed (default in Janus-Pro)
        self.image_token_num_per_image = 576
        self.img_size = 384
        self.patch_size = 16

        self.image_start_id = self.tokenizer.vocab.get("<begin_of_image>")
        self.image_end_id = self.tokenizer.vocab.get("<end_of_image>")
        self.image_id = self.tokenizer.vocab.get("<image_placeholder>")
        self.pad_id = self.tokenizer.vocab.get("<｜▁pad▁｜>")

        for k, v in self.config.generation_config.items():
            setattr(self, k, v) 
        

    def on_test_epoch_start(self):
        self.model.eval() 
    

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        try: 
            prompt_lists_by_index = [[] for _ in range(6)]
            final_path_lists_by_index = [[] for _ in range(6)]
            skip_flags = [[] for _ in range(6)]

            for sample in batch:
                item_id = sample['item_id']
                category = sample['category'] # TODO
                sub_category = sample['sub_category']
                base_prompts = sample['long_prompt']
                negative_prompts = sample['negative_long_prompt']
                perturbed_methods = sample['perturbed_method']

                for i in range(3): # = 3 perturbation type
                    for prompt_type, prompts, offset in zip(
                        ["base", "negative"],
                        [base_prompts, negative_prompts],
                        [0, 3]):
                        
                        save_path = os.path.join(self.config.save_path, 
                                                    f"{prompt_type}", 
                                                    category, 
                                                    item_id)
                        os.makedirs(save_path, exist_ok=True)
                        # save_json(save_path, "metadata.jsonl", sample)
                        save_json(save_path, "metadata", sample)
                        
                        prompt = prompts[i]
                        p_type = perturbed_methods[i]
                        idx = i + offset  # index from 0-5

                        if prompt == "" or prompt is None:
                            skip_flags[idx].append(True)
                            continue

                        formatted_prompt = get_prompt(self.processor, prompt)
                        fname = f"{i:02d}.png" 
                        # fname = get_fname(i, p_type, sub_category) # use number or p_type as a fname 
                        final_path = os.path.join(save_path, fname)

                        if os.path.exists(final_path):
                            skip_flags[idx].append(True)
                            continue

                        prompt_lists_by_index[idx].append(formatted_prompt)
                        final_path_lists_by_index[idx].append(final_path)
                        skip_flags[idx].append(False)

            for i in range(6):  # 0~5: base + negative
                if len(prompt_lists_by_index[i]) == 0:
                    continue  # Nothing to generate

                self.generate_image(
                    prompt_list=prompt_lists_by_index[i],
                    save_path_list=final_path_lists_by_index[i],
                    seed=self.config.seed_list[i % 3]  # repeat 0,1,2 for base/negative
                )

        except Exception as e:
            print(f"Error in test_step: {e}")
            traceback.print_exc()


    @torch.inference_mode()
    def generate_image(
        self,
        prompt_list: List,
        save_path_list: List,
        seed: int = None,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        set_seed(seed)

        batch_size = len(prompt_list) 
        input_ids_list = []
        max_len = 0 # for padding

        for prompt in prompt_list:
            input_ids = self.processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)

            max_len = max(max_len, len(input_ids))
            input_ids_list.append(input_ids)

        tokens = torch.zeros((batch_size*2, max_len), dtype=torch.int).to(self.device)
        attention_masks = torch.ones((batch_size*2, max_len), dtype=torch.long).to(self.device)
    
        for i in range(batch_size*2):
            pad_len = max_len - len(input_ids_list[i//2])
            tokens[i, pad_len:] = input_ids_list[i//2]
            tokens[i, :pad_len] = self.processor.pad_id
            attention_masks[i, :pad_len] = 0
            if i % 2 != 0:
                tokens[i, pad_len+1:-1] = self.processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)


        # Generate image tokens
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

            next_token = torch.multinomial(probs, num_samples=1) # torch.Size([1, 1]) / tensor([[4521]], device='cuda:0')
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

            new_mask = torch.ones((attention_masks.shape[0], 1), dtype=attention_masks.dtype, device=attention_masks.device)  
            attention_masks = torch.cat([attention_masks, new_mask], dim=1)

        # Decode generated tokens
        dec = self.model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[batch_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        
        visual_img = np.zeros((batch_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec # (1, 384, 384, 3)

        # Save generated images
        if self.parallel_size == 1:
            # batch_size > 1 
            for inner_idx, image in enumerate(visual_img):
                try:
                    Image.fromarray(image).save(save_path_list[inner_idx])
                    
                # 이 경우는 거의 발생하지 않으므로 미고려
                except OSError:
                    idx_in_path = save_path_list[inner_idx].split("_")[1] # 01.png
                    alternative_path = f"longprompt_{idx_in_path}"
                    Image.fromarray(image).save(alternative_path)

        else:
            raise NotImplementedError("parallel_size > 1, not supported.")


    # No def on_test_epoch_end(self) needed.
 

def get_dataloader(config):
    dataset = BaseDataset(fpath=config.data_path,
                          s_idx=config.s_idx, 
                          e_idx=config.e_idx)
    dataloader = DataLoader(dataset,
                            collate_fn=lambda batch: batch, 
                            batch_size=config.batch_size,
                            num_workers=config.num_workers, 
                            pin_memory=True,
                            drop_last=False)
        
    return dataloader


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    vl_chat_processor, tokenizer, model = get_model(model_path=config.model_path, cache_dir=config.cache_dir)
    assert len(config.seed_list) == 3, "Please set 3 seeds for 3 perturbation types." 

    if config.ckpt_path is not None:
        print("# Load model with checkpoint.")
        lora_config = get_lora_config(config.ckpt_path)

        model.language_model = get_peft_model(model.language_model, lora_config)
        model = JanusProTestWrapper.load_from_checkpoint(checkpoint_path=config.ckpt_path, 
                                                        config=config,
                                                        model=model,
                                                        tokenizer=tokenizer,
                                                        processor=vl_chat_processor,
                                                        strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    else:
        print("# Load base model.")
        model = JanusProTestWrapper(config=config,
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    processor=vl_chat_processor)

    # world_size = torch.cuda.device_count()
    trainer = get_eval_trainer(device, config.world_size)
    dataloader = get_dataloader(config)

    trainer.test(model, dataloaders=dataloader)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/step3.yaml")
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)
    
    main(config)