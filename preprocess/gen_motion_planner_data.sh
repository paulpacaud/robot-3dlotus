#!/bin/bash
#SBATCH --job-name=gen_pcd
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out
#SBATCH -p willow
#SBATCH -A willow

set -x
set -e

module purge
pwd; hostname; date

cd $HOME/Projects/robot-3dlotus

. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gembench

export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

#python preprocess/gen_motion_planner_data.py \
#  --old_keystep_pcd_dir data/gembench/train_dataset/keysteps_bbox_pcd/seed0/voxel0.5cm \
#  --new_keystep_pcd_dir data/gembench/train_dataset/motion_keysteps_bbox_pcd/seed0/voxel0.5cm

python preprocess/gen_motion_planner_data.py \
  --old_keystep_pcd_dir data/gembench/val_dataset/keysteps_bbox_pcd/seed100/voxel0.5cm \
  --new_keystep_pcd_dir data/gembench/val_dataset/motion_keysteps_bbox_pcd/seed100/voxel0.5cm \
