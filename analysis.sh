#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:a100:6
#SBATCH --mem=150G

python run_analysis.py