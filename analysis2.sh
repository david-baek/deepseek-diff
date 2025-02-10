#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=100G

python run_wait_analysis.py