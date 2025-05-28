# Step 5: SimPO Training
# This code is based on the official SimPO implementation with trl (https://github.com/princeton-nlp/SimPO/blob/main/scripts/simpo_trainer.py) """

import os
import argparse
from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR
import pytorch_lightning as pl
from trl.trainer.utils import pad_to_length 

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils import build_config, save_config
from ospo.base import get_train_trainer, get_model
from ospo.utils_train.datamodule import SimPODataModule
from ospo.utils_train.lr_scheduler import CosineDecayWarmUpRestarts


class SimPOWrapper(pl.LightningModule):
    def __init__(self, config, model, chat_processor, image_processor, tokenizer):
        super().__init__()
        self.config=config
        self.model=model
        self.chat_processor=chat_processor
        self.image_processor=image_processor
        self.tokenizer=tokenizer

        self.train_dataset = []
        self.val_dataset = None

        self.parallel_size = self.config.model.parallel_size # 1
        self.image_token_num_per_image = self.config.model.image_token_num_per_image # 576

        # default SimPO hyperparameter 
        simpo_config = self.config['algo']
        tokenizer_config = self.config['tokenizer']

        self.loss_type = simpo_config.get('loss_type', 'sigmoid')
        self.beta = simpo_config.get('beta', 1.0)
        self.gamma_beta_ratio = simpo_config.get('gamma_beta_ratio', 0.0)
        self.label_smoothing = simpo_config.get('label_smoothing', 0.0)
        self.sft_weight = simpo_config.get('sft_weight', 0.0)

        self.label_pad_token_id = tokenizer_config.get('label_pad_token_id', -100)
        self.padding_value = self.tokenizer.pad_token_id 
        self.max_length = tokenizer_config.get('max_length', 512)
        self.max_prompt_length = tokenizer_config.get('max_prompt_length', 128)
        

    def setup(self, stage: str):
        save_config(self.logger.log_dir, self.config)
   
        self.model.train()        
        self.model.language_model.model.config.output_hidden_states = True
        
        # Convert all parameters explicitly to FP32
        if self.config.experiment.precision == 32:
            self.model = self.model.float()

        # Freeze untrainable param
        self.freeze_param()
        self.print_trainable_parameters()
        for name, param in self.model.named_parameters():
            if param.dtype == torch.long:
                print(f"Parameter {name} is of type torch.long. Converting to float32.")
                param.data = param.data.float()


    def print_trainable_parameters(self):
        print("Trainable Parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, dtype={param.dtype}")


    def on_train_start(self):
        # set train mode
        self.model.train()
        print("Training START.")

        # Double-check critical submodules explicitly (optional but recommended)
        device = self.device
        self.model.language_model.to(device)
        self.model.gen_vision_model.to(device)
        self.model.gen_head.to(device)
        self.model.vision_model.to(device)
        self.model.aligner.to(device)
        self.model.gen_aligner.to(device)
        self.model.gen_embed.to(device)

        # Sanity check: Print confirmation once per epoch (on main process only)
        if self.trainer.is_global_zero:
            print(f"\nAll modules explicitly moved to device: {device}")


    def training_step(self, batch, batch_idx):
        """ A batch is a dict. """
        preprocessed = self.preprocess_batch(batch)
        loss = self.compute_loss(inputs=preprocessed) 
        self.log_dict({'train/loss': loss,
                       'train/lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                       'train/global_step': self.global_step}, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss # simPO loss


    def on_before_optimizer_step(self, *args, **kwargs): 
        # total grad norm
        grad_norm = self.compute_total_grad_norm()

        if grad_norm is not None:  # Ensure no logging of NoneType
            self.log('train/grad_norm', grad_norm, 
                    on_step=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            print("Warning: Grad norm is None, skipping logging.")


    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.config.optimizer.init_lr,
            betas=self.config.optimizer.betas,
            weight_decay=self.config.optimizer.weight_decay,
            eps=self.config.optimizer.eps,
        )

        if self.config.optimizer.scheduler_type == 'constant':
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=self.config.experiment.max_training_steps)

        elif self.config.optimizer.scheduler_type == 'cosine':
            warmup_step = self.config.experiment.max_training_steps * self.config.experiment.warmup_ratio
            scheduler = CosineDecayWarmUpRestarts(optimizer, 
                                                warmup_iter=warmup_step, 
                                                max_iter=self.config.experiment.max_training_steps, 
                                                eta_min=self.config.optimizer.min_lr, 
                                                eta_max=self.config.optimizer.init_lr)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }

        return [optimizer], [scheduler_config]
        
        
    # https://github.com/Lightning-AI/pytorch-lightning/blob/0.9.0/pytorch_lightning/core/lightning.py#L807-L838 (OVERRIDE)
    def configure_ddp(self, model, device_ids):
        """Override DDP to track only trainable parameters (LoRA)"""
        model = DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=False,  # Prevents error with frozen parameters
        )
        model._set_static_graph()          # Fixes the computation graph

        return model


    def freeze_param(self):
        freeze = self.config.experiment.freeze

        # We assume that we only train image-generation-related modules.
        if self.config.use_lora:
            if freeze.vision_model:
                for param in self.model.vision_model.parameters():
                    param.requires_grad = False

            if freeze.aligner:
                for param in self.model.aligner.parameters():
                    param.requires_grad = False

            if freeze.gen_vision_model:
                for name, module in self.model.gen_vision_model.named_modules():
                    if isinstance(module, torch.nn.Embedding):  # nn.Embedding Layer
                        for param in module.parameters():
                            param.requires_grad = False         # Token Embedding layer is not trainable.
                    else:
                        for param in module.parameters():
                            param.requires_grad = False

            if freeze.gen_aligner:
                for param in self.model.gen_aligner.parameters():
                    param.requires_grad = False

            if freeze.gen_head:
                for param in self.model.gen_head.parameters():
                    param.requires_grad = False

            if freeze.gen_embed:
                for param in self.model.gen_embed.parameters():
                    param.requires_grad = False
            
        else:
            self.freeze() 

            if not freeze.vision_model:
                for param in self.model.vision_model.parameters():
                    param.requires_grad = True

            if not freeze.aligner:
                for param in self.model.aligner.parameters():
                    param.requires_grad = True

            if not freeze.gen_vision_model:
                for name, module in self.model.gen_vision_model.named_modules():
                    if isinstance(module, torch.nn.Embedding):  
                        for param in module.parameters():
                            param.requires_grad = False        
                    else:
                        for param in module.parameters():
                            param.requires_grad = True

            if not freeze.gen_aligner:
                for param in self.model.gen_aligner.parameters():
                    param.requires_grad = True

            if not freeze.gen_head:
                for param in self.model.gen_head.parameters():
                    param.requires_grad = True

            if not freeze.gen_embed:
                for param in self.model.gen_embed.parameters():
                    param.requires_grad = True

            if not freeze.language_model:
                for param in self.model.language_model.parameters():
                    param.requires_grad = True


    def preprocess_batch(self, batch: Tuple):
        batch_size = len(batch[0])
        item_ids, text_tokens, chosen_image_tensors, rejected_image_tensors = batch 

        # text
        batched_text_embeds=[] 
        for text_token in text_tokens:
            text_embeds = self.model.language_model.get_input_embeddings()(text_token) 
            batched_text_embeds.append(text_embeds) # e.g. torch.Size([bsz, seq_len, 4096])

        # text padding
        max_seq_len = max(x.shape[1] for x in batched_text_embeds)
        embed_dim = batched_text_embeds[0].size(-1) # 4096
        padded_batched_text_embeds = torch.zeros(batch_size, max_seq_len, embed_dim, dtype=self.model.dtype, device=self.device) 
        padded_batched_text_labels = torch.full(
            (batch_size, max_seq_len), -100, dtype=torch.long, device=self.device
        )  # initialize with padding value 

        for i, embed in enumerate(batched_text_embeds):
            seq_len = embed.shape[1]
            padded_batched_text_embeds[i, :seq_len, :] = embed 

        # image
        expected_dtype = next(self.model.gen_vision_model.parameters()).dtype # Get model parameter dtype.
        chosen_img_token_tensor_list=[]
        rejected_img_token_tensor_list=[]

        for chosen_ts, rejected_ts in zip(chosen_image_tensors, rejected_image_tensors): 
            if chosen_ts.dtype != expected_dtype:
                chosen_ts = chosen_ts.to(dtype=expected_dtype)
            
            if rejected_ts.dtype != expected_dtype:
                rejected_ts = rejected_ts.to(dtype=expected_dtype)
            
            chosen_output = self.model.gen_vision_model.encode(chosen_ts.to(self.device)) 
            rejected_output = self.model.gen_vision_model.encode(rejected_ts.to(self.device))  

            # img_tokens: [576]
            chosen_img_tokens = chosen_output[2][2]  
            rejected_img_tokens = rejected_output[2][2]  
            
            chosen_img_token_tensor_list.append(chosen_img_tokens)
            rejected_img_token_tensor_list.append(rejected_img_tokens)
        
        batched_chosen_tensors = torch.stack(chosen_img_token_tensor_list, dim=0)
        batched_rejected_tensors = torch.stack(rejected_img_token_tensor_list, dim=0) 

        # Transform tokens into embeds.
        batched_chosen_img_embeds = self.model.prepare_gen_img_embeds(batched_chosen_tensors).to(self.device)
        batched_rejected_img_embeds = self.model.prepare_gen_img_embeds(batched_rejected_tensors).to(self.device)

        preprocessed={"item_ids": item_ids} # item_ids is List.
        preprocessed["chosen_inputs_embeds"] = torch.cat([padded_batched_text_embeds, batched_chosen_img_embeds], dim=1) 
        preprocessed["chosen_attention_mask"] = torch.ones(preprocessed["chosen_inputs_embeds"].shape[:2], dtype=torch.long) 
        preprocessed["chosen_labels"] = torch.cat([padded_batched_text_labels, batched_chosen_tensors], dim=1) 
        
        preprocessed["rejected_inputs_embeds"] = torch.cat([padded_batched_text_embeds, batched_rejected_img_embeds], dim=1) 
        preprocessed["rejected_attention_mask"] = torch.ones(preprocessed["rejected_inputs_embeds"].shape[:2], dtype=torch.long) 
        preprocessed["rejected_labels"] = torch.cat([padded_batched_text_labels, batched_rejected_tensors], dim=1) 

        return preprocessed    


    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:

        concatenated_batch = {}
        max_length = max(batch["chosen_inputs_embeds"].shape[1], batch["rejected_inputs_embeds"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = self.label_pad_token_id
                elif k.endswith("_inputs_embeds"):
                    pad_value = self.padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = self.label_pad_token_id
                elif k.endswith("_inputs_embeds"):
                    pad_value = self.padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=self.device) # .to(device=self.device)

        return concatenated_batch
        

    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(self.device)
        logits = pi_logratios - self.gamma_beta_ratio

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = self.beta * policy_chosen_logps.to(self.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.device).detach()

        return losses, chosen_rewards, rejected_rewards


    def concatenated_forward(
            self, batch: Dict[str, Union[List, torch.LongTensor]]
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

            concatenated_batch = self.concatenated_inputs(batch=batch)
            len_chosen = batch["chosen_labels"].shape[0]

            outputs = self.model.language_model.model(inputs_embeds=concatenated_batch["concatenated_inputs_embeds"], 
                                                use_cache=False, 
                                                past_key_values=None) 
            
            hidden_states = outputs.hidden_states[-1]       
            all_logits = self.model.gen_head(hidden_states)       
            all_logps = self.get_batch_logps(
                all_logits,
                concatenated_batch["concatenated_labels"],
                average_log_prob=True,
            )

            chosen_logps = all_logps[:len_chosen]
            rejected_logps = all_logps[len_chosen:]

            chosen_logits = all_logits[:len_chosen]
            rejected_logits = all_logits[len_chosen:]

            chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]

            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels)
        
        
    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
    ) -> torch.FloatTensor:
        
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

        
    def get_batch_loss_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "val"] = "train",
    ):

        prefix = "val" if train_eval == "val" else "train"
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
        ) = self.concatenated_forward(batch)

        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )

        loss = losses.mean()

        if self.sft_weight > 0.0:
            policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
            chosen_labels = chosen_labels[..., 1:].clone()

            loss_func = torch.nn.CrossEntropyLoss()

            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), chosen_labels.view(-1))
            loss = self.sft_weight * sft_loss + loss

            self.log(f"{prefix}/sft_loss", sft_loss.detach().cpu(), on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margins = (chosen_rewards - rejected_rewards)

        self.log_dict({f"{prefix}/rewards/chosen": chosen_rewards.mean().cpu(),
                       f"{prefix}/rewards/rejected": rejected_rewards.mean().cpu(),
                       f"{prefix}/rewards/accuracies": reward_accuracies.mean().cpu(),
                       f"{prefix}/rewards/margins": reward_margins.mean().cpu(),
                       f"{prefix}/logps/rejected": policy_rejected_logps.detach().mean().cpu(),
                       f"{prefix}/logps/chosen": policy_chosen_logps.detach().mean().cpu(),
                       f"{prefix}/logits/rejected": policy_rejected_logits.detach().mean().cpu(),
                       f"{prefix}/logits/chosen": policy_chosen_logits.detach().mean().cpu(),
                       }, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss


    def compute_loss(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        with torch.cuda.amp.autocast():
            loss = self.get_batch_loss_metrics(inputs, train_eval="train")

        return loss
    

    def compute_total_grad_norm(self):
        if hasattr(self.trainer.model, "get_global_grad_norm"):
            grad_norm = self.trainer.model.get_global_grad_norm()
            return grad_norm if grad_norm is not None else 0.0  # Avoid NoneType errors
        else:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5 if total_norm > 0 else 0.0  # Prevent NoneType issue


    def on_train_end(self):
        print("Training END.")


def get_dataloader(config, chat_processor, image_processor, tokenizer):
    datamodule = SimPODataModule(config, chat_processor, image_processor, tokenizer) 
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

    wrapper = SimPOWrapper(config, 
                         model=model, 
                         chat_processor=chat_processor, 
                         image_processor=image_processor,
                         tokenizer=tokenizer) 
    trainer = get_train_trainer(config, device)
    
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

