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
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out

set -x
set -e

export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID; mkdir $XDG_RUNTIME_DIR; chmod 700 $XDG_RUNTIME_DIR

module purge
module load singularity
module load cuda/12.1.0
module load gcc/11.3.1
module load miniforge/24.9.0
pwd; hostname; date

cd $WORK/Projects/robot-3dlotus

source $HOME/.bashrc
conda activate gembench

sif_image=$SINGULARITY_ALLOWED_DIR/nvcuda_v2.sif
export python_bin=$HOME/.conda/envs/gembench/bin/python
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

# --------------------------------------------------------------------------
benchmark=gembench
ckpt_version=v1_shizhe
ckpt_step=150000
taskvar=light_bulb_in_peract+17
seed=100
dataset=val

# --------------------------------------------------------------------------
model=3dlotus
expr_dir=data/experiments/${benchmark}/{model}/${ckpt_version}

singularity exec --bind $HOME:$HOME,$WORK:$WORK,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${python_bin} robot3dlotus/evaluation/eval_3dlotus_single_taskvar.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} \
    --taskvar ${taskvar} \
    --seed ${seed} \
    --microstep_data_dir data/${benchmark}/${dataset}_dataset/microsteps/seed${seed} \
    --save_obs_outs_dir $expr_dir/records/seed${seed}/${taskvar} \
    --taskvars_instructions_file assets/${benchmark}/taskvars_instructions_new.json \
    --record_video --video_dir $expr_dir/records/seed${seed}/${taskvar} \
    --not_include_robot_cameras --video_rotate_cam \
    --enable_flashattn