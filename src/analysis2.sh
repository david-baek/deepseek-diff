#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G

python run_wait_analysis.py --save_name qwen-1.5b --target_name wait
python run_wait_analysis.py --save_name qwen-7b --target_name wait
python run_wait_analysis.py --save_name qwen-14b --target_name wait


python run_wait_analysis.py --save_name qwen-1.5b --target_name deductive
python run_wait_analysis.py --save_name qwen-7b --target_name deductive
python run_wait_analysis.py --save_name qwen-14b --target_name deductive