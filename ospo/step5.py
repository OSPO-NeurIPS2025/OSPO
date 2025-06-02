# This code is based on the official SimPO implementation with trl (https://github.com/princeton-nlp/SimPO/blob/main/scripts/simpo_trainer.py) """

import os
import argparse
import torch
import pytorch_lightning as pl

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.dataclass.datamodule import TrainDataModule
from ospo.wrapper import JanusProTrainWrapper
from ospo.utils.common import build_config
from ospo.utils.train import get_trainer
from ospo.utils.model import get_model


def get_dataloader(config, chat_processor, image_processor, tokenizer):
    datamodule = TrainDataModule(config, chat_processor, image_processor, tokenizer) 
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = None 

    return train_dataloader, val_dataloader


def main(config):
    torch.cuda.empty_cache()
    if config.base.save_path is not None:
        os.makedirs(config.base.save_path, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(config.experiment.seed, workers=True) 
    dtype = torch.bfloat16 if config.experiment.precision == 'bf16' else torch.float32
   
    model, chat_processor, image_processor, tokenizer = get_model(mode='train', dtype=dtype, config=config)
    train_dataloader, val_dataloader = get_dataloader(config, chat_processor, image_processor, tokenizer) 

    wrapper = JanusProTrainWrapper(config, 
                         model=model, 
                         chat_processor=chat_processor, 
                         image_processor=image_processor,
                         tokenizer=tokenizer) 
    trainer = get_trainer(config, device)
    
    # Train the model
    if config.base.resume is not None and os.path.exists(config.base.resume):
        print("Training resume.")
        trainer.fit(wrapper, train_dataloader, val_dataloader, ckpt_path=config.base.resume)
    else:
        trainer.fit(wrapper, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/step5.yaml")
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)    
    
    main(config)
