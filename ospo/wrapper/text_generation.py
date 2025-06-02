import os
import re
import torch
import argparse
import torch.distributed as dist
from pytorch_lightning import LightningModule, seed_everything

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.constant import *
from ospo.utils.common import save_json, save_json_ddp
from ospo.utils.processor import get_sft_format, get_processor_output, batchify
from ospo.prompt.template_element import get_prompt_element
from ospo.prompt.template_negative import get_prompt_negative
from ospo.prompt.template_dense import get_prompt_dense


class JanusProElementGenWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor  
        self.category = config.category
        self.max_len = config.max_len
        if self.max_len is None:
            if config.category == "object":
                self.max_len = 120
            elif config.category == "spatial":
                self.max_len = 40
            elif config.category == "non-spatial" or config.category == "complex":
                self.max_len = 4000
            else:
                self.max_len = 70

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

            
    @torch.inference_mode()
    def generate(self, prompt):
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
                **self.config.generation_config
                )
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

        return answer


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

 

class JanusProNegativeGenWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.image_start_id = self.tokenizer.vocab.get("<begin_of_image>")
        self.image_end_id = self.tokenizer.vocab.get("<end_of_image>")
        self.image_id = self.tokenizer.vocab.get("<image_placeholder>")
        self.pad_id = self.tokenizer.vocab.get("<｜▁pad▁｜>")

        self.output_list = []
        

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

                batch[sample_idx]['negative_prompt'] = perturbed_output
                self.output_list.append(batch[sample_idx])
            
        except Exception as e:
            print(f"Error in test_step: {e}")
            traceback.print_exc()

            
    def prepare_input_embeds(self, pair_list): 
        # pair_list: list of Tuple (sub_category, prompt, p_type)
        prepare_list = []
        for triplet in pair_list:     
            sub_category, prompt, p_type = triplet
            get_func = get_prompt_negative[sub_category]

            system_prompt, conversation = get_func(p_type, prompt)
            if system_prompt is None or conversation is None:
                print("None system_prompt or conversation")
                continue

            sft_format = get_sft_format(self.processor, system_prompt, conversation)
            prepare_list.append(get_prepare_list(self.processor, self.tokenizer, sft_format))
    
        # batchify
        batched_prepares = batchify(self.processor, self.tokenizer, prepare_list)
        batched_prepares = batched_prepares.to(self.device) 
        inputs_embeds = self.model.prepare_inputs_embeds(**batched_prepares)

        return inputs_embeds, batched_prepares


    @torch.inference_mode()
    def generate(self, input_embeds, batched_prepares):
        outputs = self.model.language_model.generate(inputs_embeds=input_embeds,
            attention_mask=batched_prepares.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **self.config.generation_config
            )

        return outputs

    def on_test_epoch_end(self):
        save_json_ddp(
            save_root=self.config.save_path,
            save_name='negative_prompt',
            world_size=self.trainer.world_size,
            save_file=self.output_list,
            rank=self.trainer.global_rank,
        )
        print("Negative prompt generation done.")



class JanusProDenseGenWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.image_start_id = self.tokenizer.vocab.get("<begin_of_image>")
        self.image_end_id = self.tokenizer.vocab.get("<end_of_image>")
        self.image_id = self.tokenizer.vocab.get("<image_placeholder>")
        self.pad_id = self.tokenizer.vocab.get("<｜▁pad▁｜>")

        self.output_list = []


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
                negative_prompt_list = sample['negative_prompt'] 

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

            
    def prepare_input_embeds(self, pair_list): 
        # pair_list: list of Tuple (sub_category, base prompt, negative prompt)
        prepare_list = []
        for triplet in pair_list:     
            sub_category, base_prompt, negative_prompt = triplet
            get_func = get_prompt_gen[sub_category]
            system_prompt, conversation = get_func(base_prompt, negative_prompt)
            
            sft_format = get_sft_format(self.processor, system_prompt, conversation)
            prepare_list.append(get_prepare_list(self.processor, self.tokenizer, sft_format))

        # batchify
        batched_prepares = batchify(self.processor, self.tokenizer, prepare_list)
        batched_prepares = batched_prepares.to(self.device) 

        inputs_embeds = self.model.prepare_inputs_embeds(**batched_prepares)

        return inputs_embeds, batched_prepares


    @torch.inference_mode()
    def generate(self, input_embeds, batched_prepares):
        outputs = self.model.language_model.generate(inputs_embeds=input_embeds,
            attention_mask=batched_prepares.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **self.config.generation_config
            )

        return outputs


    def on_test_epoch_end(self):
        save_json_ddp(
            save_root=self.config.save_path,
            save_name='long_prompt',
            world_size=self.trainer.world_size,
            save_file=self.output_list,
            rank=self.trainer.global_rank,
        )
        print("Densification done.")
