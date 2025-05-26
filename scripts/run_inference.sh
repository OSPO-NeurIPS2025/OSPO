# make sure current folder is OSPO

python ospo/inference.py \
    --model_path ./checkpoints/janus-pro-7B \
    --ckpt_path ./checkpoints/ospo-epoch1.ckpt \
    --save_path ./results \
    --world_size 1 \
    --overrides # erase if not used
