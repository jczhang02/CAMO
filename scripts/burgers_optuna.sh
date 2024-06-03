#!/bin/bash
#SBATCH --job-name=optuna_Burgers_1D
#SBATCH --partition=local
#SBATCH --output=/zhangchrai23/logs/optuna_Burgers_1D.out

# hyperparameter optimization using Optuna
python src/train.py \
	experiment=burgers_optuna \
	trainer.deterministic=True \
	task_name="Optuna_Burgers"
