# make sure current folder is OSPO
# bash script/run.sh

CUDA_VISIBLE_DEVICES=0,1 

python ospo/step1.py --category object
python ospo/step1.py --category color
python ospo/step1.py --category shape
python ospo/step1.py --category texture
python ospo/step1.py --category spatial
python ospo/step1.py --category non-spatial
python ospo/step1.py --category complex
python ospo/step2.py
python ospo/step3.py 
python ospo/step4.py 
python ospo/step5.py 
