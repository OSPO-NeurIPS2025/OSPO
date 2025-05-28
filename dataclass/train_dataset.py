# README: Dataset class only includes CPU operation.

import os
import torch
import random
from PIL import Image
from typing import Dict
from torch.utils.data import Dataset

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils import read_json
from ospo.base import get_prompt


class PreferenceDataset(Dataset):
    def __init__(self, seed, data_path, 
                 chat_processor, image_processor, tokenizer, 
                 sampling_rate=1.0, num_samples=None): 
        
        self.chat_processor = chat_processor
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        # load data
        self.dataset = read_json(data_path)

        # Apply num_samples if specified
        if num_samples is not None:
            assert num_samples > 0, "num_samples must be greater than 0"
            assert num_samples <= len(self.dataset), "num_samples cannot exceed dataset size"

            # Deterministic sampling
            rng = random.Random(seed)
            indices = rng.sample(range(len(self.dataset)), num_samples)
            self.dataset = [self.dataset[i] for i in indices]

        elif sampling_rate != 1.0:
            self.total_length = int(len(self.dataset) * sampling_rate)
            assert self.total_length > 0, "Dataset size must be bigger than 1."
            self.dataset = self.dataset[:self.total_length] # from the front

        self.total_length = len(self.dataset)
        print(f"Total length of data: {self.total_length}")

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        return self.decode(example)

    def collate_fn(self, batch): 
        item_ids, text_tokens, chosen_image_tensors, rejected_image_tensors = zip(*batch)
        return list(item_ids), list(text_tokens), list(chosen_image_tensors), list(rejected_image_tensors)

    # custom functions

    def get_text_token(self, text):
        parallel_size=1 
        prompt = get_prompt(self.chat_processor, text)
        text_input_ids = self.tokenizer.encode(prompt)
        text_input_ids = torch.LongTensor(text_input_ids) # e.g. torch.Size([18])

        text_tokens = torch.zeros((parallel_size, len(text_input_ids)), dtype=torch.int) 
        for i in range(parallel_size):
            text_tokens[i, :] = text_input_ids 

        return text_tokens

    def get_image_tensor(self, img_path: str):
        image = Image.open(img_path)
        image_tensor = self.image_processor([image])
        image_tensor = image_tensor['pixel_values']  # e.g. torch.Size([1, 3, 384, 384])
       
        return image_tensor

    def decode(self, example: Dict):
        if "prompt" not in example.keys() or "chosen" not in example.keys() or "rejected" not in example.keys():
            raise ValueError(
                    f"Could not format example as dialogue for SimPO task!\nThis example only has {example.keys()} keys.\n"
                )
        
        item_id = example["item_id"]        
        text_token = self.get_text_token(example["prompt"]) 
        chosen_image_tensor = self.get_image_tensor(example["chosen"])
        rejected_image_tensor = self.get_image_tensor(example["rejected"])

        return item_id, text_token, chosen_image_tensor, rejected_image_tensor
