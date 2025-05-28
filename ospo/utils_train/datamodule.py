
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from dataclass.train_dataset import PreferenceDataset


class SimPODataModule(pl.LightningDataModule):
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