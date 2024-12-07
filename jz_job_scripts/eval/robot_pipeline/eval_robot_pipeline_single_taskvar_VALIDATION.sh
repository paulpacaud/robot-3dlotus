#!/bin/bash
#SBATCH --job-name=evRP
#SBATCH -A hjx@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
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
benchmark=peract
model=3dlotusplus
ckpt_version=v2_fine
taskvar=light_bulb_in_peract+19
seed=100
dataset=val
pc_label_type=fine
run_step=1

expr_dir=data/experiments/${benchmark}/${model}/${ckpt_version}

for ckpt_step in {100000..180000..10000}
do
singularity exec --bind $HOME:$HOME,$WORK:$WORK,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${python_bin} robot3dlotus/evaluation/eval_robot_pipeline_single_taskvar.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} \
    --taskvar ${taskvar} \
    --seed ${seed} \
    --microstep_data_dir data/${benchmark}/${dataset}_dataset/microsteps/seed${seed} \
    --full_gt \
    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline_gt.yaml \
    --gt_og_label_file assets/${benchmark}/taskvars_target_label_zrange_${benchmark}.json \
    --gt_plan_file prompts/rlbench/${benchmark}/in_context_examples_${dataset}.txt \
    --pc_label_type ${pc_label_type} --run_action_step ${run_step} \
    --enable_flashattn
done
