""" README: Step2. Prompt Densification """

import os
import re
import argparse
import traceback
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule
from peft import get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.prompt.template_dense import get_prompt_dense
from ospo.base import get_model, get_lora_config, get_eval_trainer, get_sft_format, get_prepare_list, batchify
from ospo.utils import save_json, set_seed, build_config
from data.gen_dataset import BaseDataset


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

        self.output_list = []

        # self.get_prompt_function_negative = get_prompt_negative
        self.get_prompt_function_dense = get_prompt_dense
        

    def on_test_epoch_start(self):
        self.model.eval() 
    

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        try: 
            all_outputs_by_index = [[] for _ in range(3)]  # 3 perturbation type
            
            # 1. group by perturbation index
            skip_flags = [[] for _ in range(3)]  
            grouped_pair_lists = [[] for _ in range(3)]
            
            for sample in batch:
                prompt = sample['prompt']
                sub_category = sample['sub_category']
                negative_prompt_list = sample['perturbed_prompt'] 

                for i, negative_prompt in enumerate(negative_prompt_list):
                    # Triplet: category, positive(base), negative
                    if negative_prompt == "":
                        skip_flags[i].append(True)
                        grouped_pair_lists[i].append(("", "", ""))  # Dummy triplet
                    else:
                        skip_flags[i].append(False)
                        grouped_pair_lists[i].append((sub_category, prompt, negative_prompt))


            # 2. generate per group with fixed seed
            for i, pair_list in enumerate(grouped_pair_lists):
                set_seed(self.config.seed_list[i])

                input_embeds, batched_prepares = self.prepare_input_embeds(pair_list)
                outputs = self.generate(input_embeds, batched_prepares)
                all_outputs_by_index[i] = outputs


            # 3. regroup to sample-wise format
            def post_process(raw_output):
                base_long = re.search(r"Step 2\. Prompt 1 Dense: (.+)", raw_output)
                negative_long = re.search(r"Step 4\. Prompt 2 Dense: (.+)", raw_output)

                base_long_prompt = base_long.group(1) if base_long else ""
                negative_long_prompt = negative_long.group(1) if negative_long else ""
                
                return base_long_prompt, negative_long_prompt


            for sample_idx in range(len(batch)): # = batch_size
                base_long_output = []
                negative_long_output = []

                for i in range(3):
                    if skip_flags[i][sample_idx]: 
                        base_long_prompt = ""
                        negative_long_prompt = ""
                    else:
                        output = all_outputs_by_index[i][sample_idx]
                        answer = self.tokenizer.decode(output.cpu().tolist(), skip_special_tokens=True)
                        base_long_prompt, negative_long_prompt = post_process(answer)

                        # post-process
                        if "Step 1." in base_long_prompt:
                            base_long_prompt = ""
                        if "Step 1." in negative_long_prompt:
                            negative_long_prompt = ""

                    base_long_output.append(base_long_prompt)
                    negative_long_output.append(negative_long_prompt)

                batch[sample_idx]['long_prompt'] = base_long_output
                batch[sample_idx]['negative_long_prompt'] = negative_long_output

                self.output_list.append(batch[sample_idx]) 
        
            
        except Exception as e:
            print(f"Error in test_step: {e}")
            traceback.print_exc()

            
    def prepare_input_embeds(self, pair_list): # pair_list: list of Tuple (sub_category, base prompt, negative prompt)
        prepare_list = []
        for triplet in pair_list:     
            sub_category, base_prompt, negative_prompt = triplet
            get_func = self.get_prompt_function_dense[sub_category]
            system_prompt, conversation = get_func(base_prompt, negative_prompt)
            
            sft_format = get_sft_format(self.processor, conversation, system_prompt)
            prepare_list.append(get_prepare_list(self.processor, self.tokenizer, sft_format))

        # batchify
        batched_prepares = batchify(self.processor, self.tokenizer, prepare_list)
        batched_prepares = batched_prepares.to(self.device) 

        inputs_embeds = self.model.prepare_inputs_embeds(**batched_prepares)

        return inputs_embeds, batched_prepares


    @torch.inference_mode()
    def generate(self, input_embeds, batched_prepares):
        generation_config = {
            'do_sample': self.config.generation_config.do_sample,
            'num_beams': self.config.generation_config.num_beams,
            'temperature': self.config.generation_config.temperature,
            'top_p': self.config.generation_config.top_p,
            'max_new_tokens': self.config.generation_config.max_new_tokens,
        }

        outputs = self.model.language_model.generate(inputs_embeds=input_embeds,
            attention_mask=batched_prepares.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **generation_config
            )

        return outputs


    def on_test_epoch_end(self):
        os.makedirs(self.config.save_path, exist_ok=True)
        fname = 'long_prompt.json'
        if self.config.s_idx is not None or self.config.e_idx is not None:
            fname = f'long_prompt_s_idx_{self.config.s_idx}_e_idx_{self.config.e_idx}.json'

        if self.trainer.world_size > 1:
            gathered_output_list = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(gathered_output_list, self.output_list)

            if self.trainer.global_rank == 0:
                seen = set()
                output_list = []
                for gathered_output in gathered_output_list:
                    for sample in gathered_output:
                        item_id = sample['item_id']
                        if item_id in seen:
                            continue
                        seen.add(item_id)
                        output_list.append(sample)

                sorted_output = sorted(output_list, key=lambda x: int(x["item_id"]))
                save_json(self.config.save_path, fname, sorted_output)
                
        else:
            save_json(self.config.save_path, fname, self.output_list)
 

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
                                    processor=vl_chat_processor, 
                                    config=config)

    # automatically set world_size
    # world_size = torch.cuda.device_count()
    trainer = get_eval_trainer(device, config.world_size)
    dataloader = get_dataloader(config)

    trainer.test(model, dataloaders=dataloader)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=None)
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)
    
    main(config)