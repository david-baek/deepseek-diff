#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G


python parallelogram.py --n_comp 2 --save_name qwen-14b
python parallelogram.py --n_comp 5 --save_name qwen-14b
python parallelogram.py --n_comp 10 --save_name qwen-14b
python parallelogram.py --n_comp 20 --save_name qwen-14b

python parallelogram.py --n_comp 2 --save_name qwen-1.5b
python parallelogram.py --n_comp 5 --save_name qwen-1.5b
python parallelogram.py --n_comp 10 --save_name qwen-1.5b
python parallelogram.py --n_comp 20 --save_name qwen-1.5b

python parallelogram.py --n_comp 2 --save_name qwen-7b
python parallelogram.py --n_comp 5 --save_name qwen-7b
python parallelogram.py --n_comp 10 --save_name qwen-7b
python parallelogram.py --n_comp 20 --save_name qwen-7b
