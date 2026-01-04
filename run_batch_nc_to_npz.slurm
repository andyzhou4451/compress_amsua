#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 2:00:00




# 你的环境
#source ~/.bashrc
#conda activate amsua



python batch_nc_to_npz.py \

