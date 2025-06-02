
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary

def get_trainer(device, world_size):    
    trainer = Trainer(
        accelerator=device,
        devices=world_size,
        strategy="ddp",
        max_epochs=1, 
        precision="bf16",
        callbacks=[ModelSummary(max_depth=2)],
    )

    return trainer