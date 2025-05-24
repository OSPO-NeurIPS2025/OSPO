# make sure current folder is OSPO

python ospo/inference.py \
    --model_path # Your Janus-pro-8B path
    --ckpt_path # OSPO ckpt path
    --save_path # Save path
    --world_size 1
    --overrides # erase if not used
