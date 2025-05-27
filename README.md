# OSPO
This repository contains the official implementation for the paper: **Object-centric Self-improving Preference Optimization for Text-to-Image Generation**.

![Framework](./assets/Framework.png)

## About
**OSPO** is a self-improving prefernece optimization framework for compositional text-to-image generation, allowing an MLLM to improve its fine-grained image generation capability without needing any data or external model. 


## Setup
1. Create Conda Environment
```bash
conda create -n ospo python=3.8 -y
conda activate ospo
```

2. Clone this repository
```bash
git clone https://github.com/OSPO-NeurIPS2025/OSPO.git
cd OSPO
```

3. Install Dependencies
```bash
# We use pytorch 2.0.1 CUDA 11.7 Version
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install Dependencies
pip install -r requirements.txt
```

4. Download Janus-Pro-7B Model
- Janus-Pro-7B: 🤗[HuggingFace](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
- Make `./checkpoints` directory and place the Model in the `./checkpoints` directory.
- So the Model folder path should be `./checkpoints/Janus-Pro-7B`


## Inference
1. For inference, please download the weights from the following link:
- Janus-Pro-7B trained by **OSPO** Framework: [Google Drive](https://drive.google.com/file/d/1AI42LfljJ5nl2YZ-AVuD0sziYs0KB_yx/view?usp=sharing)
2. Place the checkpoint in the `./checkpoints` directory.
3. So the checkpoint path should be `./checkpoints/ospo-epoch1.ckpt`

4. Run inference
```bash
bash scripts/run_inference.sh
```
Results will be saved in `./results`


## Acknowledgement
We thank the authors of [Janus-Pro](https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus-pro), for making their code available.


## TODO
- [ ] Release the OSPO framework code.
- [ ] Release the train dataset.
- [x] Release the model checkpoint.
- [x] Release the inference code for text-to-image generation.
- [ ] Release the evaluation code.
