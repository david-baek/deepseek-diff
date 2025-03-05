#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G

python run_wait_analysis.py --save_name qwen-1.5b --target_name alternative
python run_wait_analysis.py --save_name qwen-7b --target_name alternative
python run_wait_analysis.py --save_name qwen-14b --target_name alternative


python run_wait_analysis.py --save_name qwen-1.5b --target_name contrastive
python run_wait_analysis.py --save_name qwen-7b --target_name contrastive
python run_wait_analysis.py --save_name qwen-14b --target_name contrastive