import os
import argparse

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

# TODO
from ospo.wrapper import *
from ospo.datamodule import *


def get_dataloader(config, step:int=None):
    if step == 5:    
        datamodule = TrainDataModule(config, chat_processor, image_processor, tokenizer) 
        datamodule.setup()
        train_dataloader = datamodule.train_dataloader()
        return train_dataloader

    else:
        datamodule = GenerationDataModule(config, step)
        gen_dataloader = datamodule.gen_dataloader()
        return gen_dataloader


def main(config):
    
    # Step 1
    # Step 2
    # Step 3
    # Step 4
    # Step 5


if __main__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO
    parser.add_argument("--cfg_path", type=str, default="configs/step1.yaml")
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)
    
    main(config)