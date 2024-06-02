#!/bin/bash
#SBATCH --job-name=Aimup
#SBATCH --partition=local
#SBATCH --output=/zhangchrai23/logs/aimup.out
aim up --host 0.0.0.0 -p 6007
