#!/bin/bash
#SBATCH --job-name=Burgers_1D_exps_lr_nhead
#SBATCH --partition=local
#SBATCH --output=/zhangchrai23/logs/slurm_Burgers_1D.out

# Evalution the impact of different settings of lr and nhead.

python src/train.py \
	experiment=burgers \
	model.net.input_encoder_config.nhead=4 \
	model.optimizer.lr=1.0e-3,1.0e-4,3.5e-4,8.0e-4 \
	trainer.deterministic=True \
	task_name="Burgers_1D_exps_lr" \
	-m
