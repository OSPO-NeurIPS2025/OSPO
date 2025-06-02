import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json
from dataclass import PreferenceDataset, BaseDataset


class TrainDataModule(pl.LightningDataModule):
    def __init__(self, config, chat_processor, image_processor, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.chat_processor = chat_processor
        self.image_processor = image_processor

        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        self.train_dataset = PreferenceDataset(
                                        seed=self.config.experiment.seed,
                                        data_path=self.config.dataset.train.data_path,
                                        chat_processor=self.chat_processor,
                                        image_processor=self.image_processor,
                                        tokenizer=self.tokenizer,
                                        num_samples=self.config.dataset.train.num_samples,
                                        # omit sampling_rate
                                        )
        
        # TODO: self.val_dataset

    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(
                    dataset=self.train_dataset, 
                    shuffle=True,
                    collate_fn=self.train_dataset.collate_fn,            # lambda batch: batch, 
                    batch_size=self.config.dataset.train.batch_size,     # Batch size per process
                    num_workers=self.config.dataset.train.num_workers,   # Number of data loading workers
                    )
    
    def val_dataloader(self):
        return None # TODO


# TODO: datapath
class GenerationDataModule(pl.LightningDataModule):
    def __init__(self, config, step:int=1): # step in framework
        if step not in [1, 2, 3, 4]: 
            raise ValueError("Step must be one of [1, 2, 3, 4].")
        self.config = config

        # Element/Base prompt generation
        if step == 1:
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
            self.dataset = list(range(max_len)) # dummy data

        # Dense/Negative prompt generation and Image generation
        elif step == 2 or step == 3:
            self.dataset = BaseDataset(fpath=config.data_path,
                                    s_idx=config.s_idx, 
                                    e_idx=config.e_idx)
        
        # Question/Answer generation
        elif step == 4:
            self.dataset = read_json(config.data_path) 


    def gen_dataloader(self):
        return DataLoader(self.dataset, 
                        batch_size=self.config.batch_size, 
                        collate_fn=lambda batch: batch,
                        num_workers=self.config.num_workers,
                        pin_memory=True,
                        drop_last=False,
                        shuffle=False)
        