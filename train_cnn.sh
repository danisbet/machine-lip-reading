#!/bin/bash
# Job name:
#SBATCH --job-name=lipreading-train
#
# Account:
#SBATCH --account=fc_mlsec
#
# Wall clock limit:
#SBATCH --time=23:59:00
#
# Partition:
#SBATCH --partition=savio2_1080ti
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
# Memory:
#SBATCH --mem-per-cpu=30G
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=alex_vlissidis@berkeley.edu
#
## Command(s) to run:
source deactivate
module purge
module load tensorflow/1.5.0-py35-pip-gpu
module unload cudnn/7.1
module load cudnn/7.0.5
python train_cnn.py

