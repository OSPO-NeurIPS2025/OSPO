""" README: Step2. Negative Prompt Generation """

import os
import argparse
import traceback
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule
from peft import get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.prompt.template_negative import get_prompt_negative
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

        self.get_prompt_function_negative = get_prompt_negative
        # self.get_prompt_function_dense = get_prompt_dense
        

    def on_test_epoch_start(self):
        self.model.eval() 
    

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        try: 
            all_outputs_by_index = [[] for _ in range(3)]  # 3 perturbation type
            
            # 1. group by perturbation index
            grouped_pair_lists = [[] for _ in range(3)]
            for sample in batch:
                prompt = sample['prompt']
                sub_category = sample['sub_category']
                perturbed_methods = sample['perturbed_method']

                for i, p_type in enumerate(perturbed_methods):
                    grouped_pair_lists[i].append((sub_category, prompt, p_type))

            # 2. generate per group with fixed seed
            for i, pair_list in enumerate(grouped_pair_lists):
                set_seed(self.config.seed_list[i])

                input_embeds, batched_prepares = self.prepare_input_embeds(pair_list)
                outputs = self.generate(input_embeds, batched_prepares)
                all_outputs_by_index[i] = outputs

            # 3. regroup to sample-wise format
            for sample_idx in range(len(batch)): # = batch_size
                perturbed_output = []

                for i in range(3):
                    output = all_outputs_by_index[i][sample_idx]
                    answer = self.tokenizer.decode(output.cpu().tolist(), skip_special_tokens=True)
                    answer_output = answer.split("Contrastive Prompt: ")[-1].strip()

                    # post-process 1
                    if "<pos>" in answer_output:
                        answer_output = answer_output.replace("<pos>", "").strip()

                    # post-process 2
                    if "Step 1." in answer_output:
                        answer_output = ""

                    perturbed_output.append(answer_output)

                batch[sample_idx]['perturbed_prompt'] = perturbed_output
            
        except Exception as e:
            print(f"Error in test_step: {e}")
            traceback.print_exc()

            
    def prepare_input_embeds(self, pair_list): # pair_list: list of Tuple (sub_category, prompt, p_type)
        prepare_list = []
        for triplet in pair_list:     
            sub_category, prompt, p_type = triplet
            get_func = self.get_prompt_function_negative[sub_category]

            system_prompt, conversation = get_func(p_type, prompt)
            if system_prompt is None or conversation is None:
                print("None system_prompt or conversation")
                continue

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
        fname = 'negative_prompt.json'
        if self.config.s_idx is not None or self.config.e_idx is not None:
            fname = f'negative_prompt_s_idx_{self.config.s_idx}_e_idx_{self.config.e_idx}.json'

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

    # world_size = torch.cuda.device_count()
    trainer = get_eval_trainer(device, config.world_size)
    dataloader = get_dataloader(config)

    trainer.test(model, dataloaders=dataloader)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/step2.yaml")
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)
    
    main(config)