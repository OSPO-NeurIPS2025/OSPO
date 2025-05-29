# Step 4: Self-VQA for Preference Pair Selection

import os
import argparse
import numpy as np
from glob import glob
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, seed_everything
from peft import get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils import read_json, save_json, build_config
from ospo.base import get_model, get_lora_config, get_eval_trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class JanusVQAWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor):
        super().__init__()
        self.config=config
        self.model=model
        self.tokenizer=tokenizer
        self.processor=processor

        self.output_list = []
        # Token IDs
        self.yes_ids = [self.tokenizer("yes", add_special_tokens=False).input_ids[-1],
                        self.tokenizer("Yes", add_special_tokens=False).input_ids[-1]]
        self.no_ids  = [self.tokenizer("no", add_special_tokens=False).input_ids[-1],
                        self.tokenizer("No", add_special_tokens=False).input_ids[-1]]


    def on_test_epoch_start(self):
        self.model.eval() 


    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        questions_batched = [sample['question'] for sample in batch]  
        base_paths_batched, negative_paths_batched = [], []

        for sample in batch:
            base_paths = sorted(glob(os.path.join(self.config.image_path, 'base', sample['category'], sample['item_id'], '*.png')))
            negative_paths = sorted(glob(os.path.join(self.config.image_path, 'negative', sample['category'], sample['item_id'], '*.png')))
            base_paths_batched.append(base_paths)
            negative_paths_batched.append(negative_paths)

        # # For base, (chosen candidate)
        # base_all_convs, base_all_imgs, base_mapping = self.get_mapping(base_paths_batched, questions_batched)
        # base_logits = self.forward(base_all_convs, base_all_imgs)
        # base_img_metadata_batched = self.get_score(base_paths_batched, questions_batched, base_mapping, base_logits)

        # # For negative, (rejected candidate)
        # negative_all_convs, negative_all_imgs, negative_mapping = self.get_mapping(negative_paths_batched, questions_batched)
        # negative_logits = self.forward(negative_all_convs, negative_all_imgs)
        # negative_img_metadata_batched = self.get_score(negative_paths_batched, questions_batched, negative_mapping, negative_logits)

        base_img_metadata_batched = self.get_score_single(base_paths_batched, questions_batched)
        negative_img_metadata_batched = self.get_score_single(negative_paths_batched, questions_batched)

        # Select pair
        self.select_pair(batch, base_img_metadata_batched, negative_img_metadata_batched)


    def get_mapping(self, img_paths_batched, questions_batched):
        # (sample_idx(i), img_idx, q_idx)
        batch_size = len(img_paths_batched)
        all_convs, all_imgs, mapping = [], [], []
        
        for i in range(batch_size):
            for img_idx, img_path in enumerate(img_paths_batched[i]):
                img = Image.open(img_path)
                for q_idx, question in enumerate(questions_batched[i]):
                    all_convs.append([
                        {"role": "<|User|>",
                         "content": f"<image_placeholder>\n{question} Please answer 'yes' or 'no' without explanation.",
                         "images": [img]},
                        {"role": "<|Assistant|>", "content": ""}
                    ])
                    all_imgs.append([img])
                    mapping.append((i, img_idx, q_idx))
        
        return all_convs, all_imgs, mapping


    def forward(self, all_convs, all_imgs): 
        prepare_list = []
        for convs, imgs in zip(all_convs, all_imgs):
            prepare = self.processor.process_one(
                conversations=convs,
                images=imgs,
                force_batchify=True
            )
            prepare_list.append(prepare)
        
        # forward 
        with torch.no_grad():
            batch_inputs = self.processor.batchify(prepare_list).to(self.device)
            inputs_embeds = self.model.prepare_inputs_embeds(**batch_inputs)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch_inputs.attention_mask
            )
        logits = outputs.logits  # [B_total, seq_len, vocab_size]

        return logits
    
    def build_conversations(self, img, questions):
        conversations = []
        for q in questions:
            conversations.append([
                {"role": "<|User|>",
                "content": f"<image_placeholder>\n{q} Please answer 'yes' or 'no' without explanation.",
                "images": [img]},
                {"role": "<|Assistant|>", "content": ""}
            ])
        return conversations, [[img]] * len(questions)

    def get_score_single(self, img_paths_batched, questions_batched):
        img_metadata_batched = []

        for sample_idx, img_paths in enumerate(img_paths_batched):
            sample_metadata = {}

            for img_idx, img_path in enumerate(img_paths):
                with Image.open(img_path) as img:
                    convs, imgs = self.build_conversations(img, questions_batched[sample_idx])
                    logits = self.forward_single(convs, imgs)  # shape: [num_questions, seq_len, vocab]
                    probs = torch.softmax(logits[:, -1, :], dim=-1)

                    score_sum = 0
                    answer_metadata = []
                    q_count = len(questions_batched[sample_idx])

                    for q_idx in range(q_count):
                        p_yes = max(probs[q_idx, y].item() for y in self.yes_ids)
                        p_no = max(probs[q_idx, n].item() for n in self.no_ids)

                        answer_metadata.append({
                            'p_yes': float(p_yes),
                            'p_no': float(p_no),
                            'answer': 'yes' if p_yes > p_no else ('no' if p_no > p_yes else 'tie')
                        })

                        if q_idx == q_count - 1:
                            global_score = p_yes - p_no
                        else:
                            score_sum += (p_yes - p_no)

                    local_score = score_sum / (q_count - 1)
                    prefix = 'base' if 'base' in img_path else 'negative'

                    sample_metadata[f'{prefix}_{img_idx}'] = {
                        'path': img_path,
                        'local_score': float(local_score),
                        'global_score': float(global_score),
                        'answer_metadata': answer_metadata
                    }

            img_metadata_batched.append(sample_metadata)

        return img_metadata_batched

    
    def forward_single(self, convs, imgs):
        prepare_list = []
        for conv, img in zip(convs, imgs):
            prepare = self.processor.process_one(
                conversations=conv,
                images=img,
                force_batchify=True
            )
            prepare_list.append(prepare)

        with torch.no_grad():
            batch_inputs = self.processor.batchify(prepare_list).to(self.device)
            inputs_embeds = self.model.prepare_inputs_embeds(**batch_inputs)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch_inputs.attention_mask
            )
        return outputs.logits


    def get_score(self, img_paths_batched, questions_batched, mapping, logits):
        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)

        batch_size = len(img_paths_batched)
        img_cnt_batched = [len(paths) for paths in img_paths_batched]
        question_cnt_batched = [len(questions) for questions in questions_batched]
        answer_metadata_batched = [[[None] * question_cnt_batched[i] for _ in range(img_cnt_batched[i])] for i in range(batch_size)]

        score_sum_batched = [np.zeros(n, dtype=float) for n in img_cnt_batched]
        global_score_batched = [np.zeros(n, dtype=float) for n in img_cnt_batched]
        local_score_batched = [np.zeros(n, dtype=float) for n in img_cnt_batched]

        # accumulate per sequence
        for seq_idx, (sample_idx, img_idx, q_idx) in enumerate(mapping):
            question_cnt = question_cnt_batched[sample_idx]

            p_yes = max(probs[seq_idx, y].item() for y in self.yes_ids)
            p_no  = max(probs[seq_idx, n].item() for n in self.no_ids)

            # last question
            if q_idx == question_cnt - 1:
                global_score_batched[sample_idx][img_idx] = (p_yes - p_no)

            else:
                score_sum_batched[sample_idx][img_idx] += (p_yes - p_no)

            answer_metadata_batched[sample_idx][img_idx][q_idx] = {
                'p_yes': float(p_yes),
                'p_no': float(p_no),
                'answer': 'yes' if p_yes > p_no else ('no' if p_no > p_yes else 'tie')
            }

        # compute local score per image
        img_metadata_batched = []

        for sample_idx, score_sum_by_sample in enumerate(score_sum_batched):
            len_q = question_cnt_batched[sample_idx] - 1 # except for global question
            for img_idx, score_sum_by_image in enumerate(score_sum_by_sample):
                local_score = score_sum_by_image / len_q
                local_score_batched[sample_idx][img_idx] = local_score


        # 모든 샘플에 대한 img_metadata 
        for sample_idx in range(batch_size):
            # 해당 샘플의 이미지 개수
            img_metadata = {}
            img_path_list = img_paths_batched[sample_idx]

            if 'base' in img_path_list[0]:
                prefix = 'base'
            elif 'negative' in img_path_list[0]:
                prefix = 'negative'
            else:
                raise ValueError(f"Image path does not contain 'base' or 'negative'. Image path: {img_path_list[0]}")

            for img_idx, img_path in enumerate(img_path_list):
                img_metadata[f'{prefix}_{img_idx}'] = {
                    'path': img_path,
                    'local_score': float(local_score_batched[sample_idx][img_idx]),
                    'global_score': float(global_score_batched[sample_idx][img_idx]),
                    'answer_metadata': answer_metadata_batched[sample_idx][img_idx]
                }
            img_metadata_batched.append(img_metadata)

        return img_metadata_batched


    def compute_preference_strength(self, base_img_dict, negative_img_dict):
        # For each sample, we assume that length of base and negative have same length.
        bases = [(i, base_img_dict.get(f'base_{i}')) for i in range(3) if base_img_dict.get(f'base_{i}') is not None]
        negatives = [(i, negative_img_dict.get(f'negative_{i}')) for i in range(3) if negative_img_dict.get(f'negative_{i}') is not None]
        if len(bases) == 0 or len(negatives) == 0:
            return None

        pairs = []
        for idx in range(3):  
            base = base_img_dict.get(f'base_{idx}')
            neg = negative_img_dict.get(f'negative_{idx}')

            if base is not None and neg is not None:
                local_gap = base['local_score'] - neg['local_score']
                global_gap = base['global_score'] - neg['global_score']

                # filter
                if local_gap >= 0 and global_gap >= 0:
                    pairs.append({
                        'pair_idx': idx,
                        'local_gap': local_gap,
                        'global_gap': global_gap
                    })
        if not pairs: 
            return None

        # For normalization,
        max_local_gap = max(abs(p['local_gap']) for p in pairs)
        max_global_gap = max(abs(p['global_gap']) for p in pairs)

        best_score = -np.inf
        best_pair = None

        for pair in pairs:
            norm_local = abs(pair['local_gap']) / (max_local_gap + 1e-8)
            norm_global = abs(pair['global_gap']) / (max_global_gap + 1e-8)
            preference_strength = norm_local / (norm_global + 1e-8)  # Avoid division by zero

            if preference_strength > best_score:
                best_score = preference_strength
                best_pair = pair
        if best_pair is None:
            return None
        
        chosen = base_img_dict[f'base_{best_pair["pair_idx"]}']['path']
        rejected = negative_img_dict[f'negative_{best_pair["pair_idx"]}']['path']
        score_metadata = {
            "local_gap": best_pair["local_gap"],
            "global_gap": best_pair["global_gap"],
            "preference_strength": best_score,
        }

        return (chosen, rejected, score_metadata)
    

    def select_pair(self, batch, base_metadata_batched, negative_metadata_batched):
        for sample_idx, (base_dict, negative_dict) in enumerate(zip(base_metadata_batched, negative_metadata_batched)):
            # Compute preference strength
            result = self.compute_preference_strength(base_dict, negative_dict)
            if result is None:
                continue
            else:
                chosen, rejected, score_metadata = result
            
            # Prepare training dataset
            output = {
                "item_id": batch[sample_idx]['item_id'],
                "category": batch[sample_idx]['category'],
                "sub_category": batch[sample_idx]['sub_category'],
                "question": batch[sample_idx]['question'],
                "prompt": batch[sample_idx]['prompt'],
                "chosen": chosen,
                "rejected": rejected,
                "metadata": { 
                    "score_metadata": score_metadata,
                    "base_meatadata": base_dict,
                    "negative_metadata": negative_dict
                }
            }
            self.output_list.append(output)


    def on_test_epoch_end(self):
        os.makedirs(self.config.save_path, exist_ok=True)
        fname = 'train'
        if self.config.s_idx is not None or self.config.e_idx is not None:
            fname = f'train_s_idx_{self.config.s_idx}_e_idx_{self.config.e_idx}'

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
    data_path = os.path.join(config.save_path, "vqa_prompt.json")
    dataset = read_json(data_path) 
    dataloader = DataLoader(dataset,
                            collate_fn=lambda batch: batch, 
                            batch_size=config.batch_size,
                            num_workers=config.num_workers, 
                            pin_memory=True,
                            drop_last=False)
    return dataloader
    

def main(config):

    # assert config.batch_size == 1, "Batch size must be 1 to prevent OOM."

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    seed_everything(config.seed, workers=True)

    vl_chat_processor, tokenizer, model = get_model(model_path=config.model_path, cache_dir=config.cache_dir)
    
    if config.ckpt_path is not None:
        print("# Load model with checkpoint.")
        lora_config = get_lora_config(config.ckpt_path)

        model.language_model = get_peft_model(model.language_model, lora_config)
        model = JanusVQAWrapper.load_from_checkpoint(checkpoint_path=config.ckpt_path, 
                                                        config=config,
                                                        model=model,
                                                        tokenizer=tokenizer,
                                                        processor=vl_chat_processor,
                                                        strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    else:
        print("# Load base model.")
        model = JanusVQAWrapper(config=config,
                                model=model, 
                                tokenizer=tokenizer, 
                                processor=vl_chat_processor)


    dataloader = get_dataloader(config)
    trainer = get_eval_trainer(device, config.world_size)

    trainer.test(model, dataloader)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/step4.yaml")
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)
    
    main(config)