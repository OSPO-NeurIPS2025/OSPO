
import os
import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from transformers import AutoModelForCausalLM
from peft import LoraConfig

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.processing_vlm import VLChatProcessorOutput, BatchedVLChatProcessorOutput


def get_model(model_path, cache_dir=None):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, cache_dir)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    return vl_chat_processor, tokenizer, vl_gpt


def get_lora_config(ckpt_path):
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(ckpt_config_path, "r") as file:
        ckpt_config = yaml.safe_load(file)

    # Extract LoRA config
    lora_config = LoraConfig(
        r=ckpt_config["lora"].get("lora_rank"),
        lora_alpha=ckpt_config["lora"]["lora_alpha"],
        target_modules=ckpt_config["lora"]["target_modules"],
        lora_dropout=ckpt_config["lora"]["lora_dropout"],
        modules_to_save=ckpt_config["lora"].get("modules_to_save")                     
    )
 
    return lora_config


def get_eval_trainer(device, world_size):    
    trainer = Trainer(
        accelerator=device,
        devices=world_size,
        strategy="ddp",
        max_epochs=1, 
        precision="bf16",
        callbacks=[ModelSummary(max_depth=2)],
    )

    return trainer


def get_train_trainer(args, device):    
    trainer = Trainer(
        accelerator=device,
        devices=args.world_size,
        strategy="ddp",
        max_epochs=1, 
        precision="bf16",
        callbacks=[ModelSummary(max_depth=2)],
    )

    return trainer



# --------------------------------------------------------------------------


def get_sft_format(processor, system_prompt, conversation):
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format='deepseek',
                system_prompt=system_prompt
            )
    return sft_format


def batchify(processor, tokenizer, prepare_list):
    batch_size = len(prepare_list)

    n_images = []
    seq_lens = []
    sft_format = []

    pad_id = tokenizer.vocab.get("<｜▁pad▁｜>")
    image_id = tokenizer.vocab.get("<image_placeholder>")
    image_token_num_per_image = 576

    for prepare in prepare_list:
        n_images.append(len(prepare.num_image_tokens)) # 0
        seq_lens.append(len(prepare))

    input_token_max_len = max(seq_lens)
    max_n_images = max(1, max(n_images))

    batched_input_ids = torch.full(
        (batch_size, input_token_max_len), pad_id
    ).long()  # FIXME
    batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
    batched_pixel_values = torch.zeros(
        (batch_size, max_n_images, *processor.image_processor.default_shape)
    ).float()
    batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
    batched_images_emb_mask = torch.zeros(
        (batch_size, max_n_images, image_token_num_per_image)
    ).bool()

    for i, prepare in enumerate(prepare_list):
        input_ids = prepare.input_ids
        seq_len = len(prepare)
        n_image = len(prepare.num_image_tokens)
        # left-padding
        batched_attention_mask[i, -seq_len:] = 1
        batched_input_ids[i, -seq_len:] = torch.LongTensor(input_ids)
        batched_images_seq_mask[i, -seq_len:] = input_ids == image_id

        if n_image > 0:
            batched_pixel_values[i, :n_image] = prepare.pixel_values
            for j, n_image_tokens in enumerate(prepare.num_image_tokens):
                batched_images_emb_mask[i, j, :n_image_tokens] = True

        sft_format.append(prepare.sft_format)

    batched_prepares = BatchedVLChatProcessorOutput(
        input_ids=batched_input_ids,
        attention_mask=batched_attention_mask,
        pixel_values=batched_pixel_values,
        images_seq_mask=batched_images_seq_mask,
        images_emb_mask=batched_images_emb_mask,
        sft_format=sft_format,
    )

    return batched_prepares
        
        
def get_prepare_list(processor, tokenizer, sft_format):
    image_id = tokenizer.vocab.get("<image_placeholder>")
    image_token_num_per_image = 576

    input_ids = processor.tokenizer.encode(sft_format)
    input_ids = torch.LongTensor(input_ids)    

    # without image
    image_token_mask = input_ids == image_id
    image_indices = image_token_mask.nonzero()

    images_outputs = processor.image_processor([], return_tensors="pt")
    num_image_tokens = torch.IntTensor([image_token_num_per_image] * len(image_indices))

    output = VLChatProcessorOutput(
        sft_format=sft_format,
        input_ids=input_ids,
        pixel_values=images_outputs.pixel_values,
        num_image_tokens=num_image_tokens,
    )

    return output


def get_prompt(vl_chat_processor, text):
    conversation = [
        {
            "role": "User",
            "content": text, # "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
        },
        {"role": "Assistant", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag
    return prompt


def get_fname(i, p_type, sub_category):
    if sub_category in ["attribute1_color", "attribute1_texture", "attribute1_shape", "layout2"]:
        fname = f"replace_{i+1}.png"
    elif sub_category == "non-spatial":
        fname = f"{p_type}.png" if p_type == "drop" else f"{p_type}_{i+1}.png" # replace_1.png, replace_2.png
    elif sub_category in ["complex", "attribute2", "layout1", "layout3"]:
        fname = f"{p_type}.png"
    else:
        raise ValueError(f"Unknown sub_category: {sub_category}")
    
    return fname