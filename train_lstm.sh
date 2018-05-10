#!/bin/bash
# Job name:
#SBATCH --job-name=lipreading-lstm
#
# Account:
#SBATCH --account=fc_mlsec
#
# Wall clock limit:
#SBATCH --time=23:59:00
#
# Partition:
#SBATCH --partition=savio2_bigmem
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
##SBATCH --gres=gpu:1
#
#SBATCH --cpus-per-task=2
#
##SBATCH --mem-per-cpu=31G
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
#module load tensorflow/1.5.0-py35-pip-gpu
#module unload cudnn/7.1
#module load cudnn/7.0.5
module load python/2.7
source activate cs294-129
python train_lstm_oh.py -sid 19
