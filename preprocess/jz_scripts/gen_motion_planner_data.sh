#!/bin/bash
#SBATCH --job-name=evSP
#SBATCH -A hjx@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
##SBATCH --mem=40G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out

## v100: qos_gpu-t3 (défaut), qos_gpu-t4, qos_gpu-dev
## a100: qos_gpu_a100-t3 (défaut), qos_gpu_a100-dev
## h100: qos_gpu_h100-t3 (défaut), qos_gpu_h100-dev, qos_gpu_h100-t4
##SBATCH -A uta42aa
##SBATCH -C v100-16g
##SBATCH -C v100-32g
##SBATCH --partition=gpu_p13

set -x
set -e

#export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
#mkdir $XDG_RUNTIME_DIR
#chmod 700 $XDG_RUNTIME_DIR

module purge
module load singularity
module load cuda/12.1.0
module load gcc/11.3.1
module load miniforge/24.9.0
pwd; hostname; date

cd $WORK/Projects/robot-3dlotus

source $HOME/.bashrc
conda activate gembench

python preprocess/gen_motion_planner_data_peract.py
