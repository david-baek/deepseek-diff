#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=150G

python ../crosscoder_diff/train.py