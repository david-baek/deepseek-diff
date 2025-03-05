#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G

python run_ablation.py --half_ablate 0 --frac 0.5 --target_name contrastive
python run_ablation.py --half_ablate 0 --frac 0.2 --target_name contrastive
python run_ablation.py --half_ablate 0 --frac 0.1 --target_name contrastive
python run_ablation.py --half_ablate 0 --frac 0.05 --target_name contrastive
python run_ablation.py --half_ablate 0 --frac 0.02 --target_name contrastive
python run_ablation.py --half_ablate 0 --frac 0.01 --target_name contrastive
python run_ablation.py --half_ablate 0 --frac 0.005 --target_name contrastive

python run_ablation.py --half_ablate 1 --frac 0.5 --target_name contrastive
python run_ablation.py --half_ablate 1 --frac 0.2 --target_name contrastive
python run_ablation.py --half_ablate 1 --frac 0.1 --target_name contrastive
python run_ablation.py --half_ablate 1 --frac 0.05 --target_name contrastive
python run_ablation.py --half_ablate 1 --frac 0.02 --target_name contrastive
python run_ablation.py --half_ablate 1 --frac 0.01 --target_name contrastive
python run_ablation.py --half_ablate 1 --frac 0.005 --target_name contrastive
