
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy


def get_trainer(config, device):
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=config.base.save_path, name=config.base.exp_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename="{step:06d}",
        save_top_k=-1,         
        every_n_train_steps=config.experiment.save_steps, 
    )

    if config.use_lora:
        trainer = Trainer(
            devices=config.base.world_size,
            accelerator=device,
            logger=tb_logger,
            default_root_dir=config.base.save_path,
            callbacks=[pl.callbacks.ModelSummary(max_depth=2), checkpoint_callback], 
            strategy=DDPStrategy(                                          
                find_unused_parameters=False # LoRA issue
            ),          
            log_every_n_steps=config.experiment.log_steps,
            gradient_clip_val=config.experiment.gradient_clip_val, 
            enable_checkpointing=config.experiment.enable_checkpointing,
            accumulate_grad_batches=config.experiment.gradient_accumulation_steps,
            precision="bf16" if config.experiment.precision is None or config.experiment.precision == "auto" else config.experiment.precision, #config.precision, 
            max_steps=config.experiment.max_training_steps, # or max_epochs   
            check_val_every_n_epoch=None, # no validation
            val_check_interval=config.experiment.val_steps * config.experiment.gradient_accumulation_steps, 
        )
        
    else:
        trainer = Trainer(
            devices=config.base.world_size,
            accelerator=device,
            logger=tb_logger,
            default_root_dir=config.base.save_path,
            callbacks=[pl.callbacks.ModelSummary(max_depth=2), checkpoint_callback], 
            strategy=DDPStrategy(                                          
                find_unused_parameters=False 
            ),          
            log_every_n_steps=config.experiment.log_steps,
            gradient_clip_val=config.experiment.gradient_clip_val, 
            enable_checkpointing=config.experiment.enable_checkpointing,
            accumulate_grad_batches=config.experiment.gradient_accumulation_steps,
            precision="bf16" if config.experiment.precision is None or config.experiment.precision == "auto" else config.experiment.precision, #config.precision, 
            max_steps=config.experiment.max_training_steps, # or max_epochs   
            check_val_every_n_epoch=None, # no validation
            val_check_interval=config.experiment.val_steps * config.experiment.gradient_accumulation_steps,        
        )

    return trainer
