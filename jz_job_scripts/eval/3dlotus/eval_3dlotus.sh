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

sif_image=$SINGULARITY_ALLOWED_DIR/nvcuda_v2.sif
export python_bin=$HOME/.conda/envs/gembench/bin/python
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

expr_dir=data/experiments/gembench/3dlotus/v1_shizhe
ckpt_step=150000

### validation
taskvar=open_box+0
seed=100
singularity exec --bind $HOME:$HOME,$WORK:$WORK,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${python_bin} robot3dlotus/evaluation/eval_3dlotus_single_taskvar.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} \
    --taskvar ${taskvar} \
    --num_episodes 20 --seed ${seed} \
    --microstep_data_dir data/gembench/val_dataset/microsteps/seed${seed} \
    --save_obs_outs_dir $expr_dir/records/seed${seed}/${taskvar} \
    --record_video --video_dir $expr_dir/records/seed${seed}/${taskvar} \
    --not_include_robot_cameras --video_rotate_cam \
    --enable_flashattn

#singularity exec --bind $HOME:$HOME,$WORK:$WORK,$SCRATCH:$SCRATCH --nv ${sif_image} \
#    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_simple_policy_server.py \
#    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 4 \
#    --taskvar_file assets/taskvars_train.json \
#    --seed 100 --num_demos 20 \
#    --microstep_data_dir data/gembench/val_dataset/microsteps/seed100

#singularity exec --bind $HOME:$HOME,$WORK:$WORK,$SCRATCH:$SCRATCH --nv ${sif_image} \
#    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_simple_policy.py \
#    --exp_config ${expr_dir}/logs/training_config.yaml \
#    --checkpoint ${expr_dir}/ckpts/model_step_${ckpt_step}.pt \
#    --taskvar push_button+3  --seed 100 --num_demos 20 \
#    --microstep_data_dir data/gembench/val_dataset/microsteps/seed100
##    --record_video --not_include_robot_cameras --video_rotate_cam


