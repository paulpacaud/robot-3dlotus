#!/bin/bash
#SBATCH --job-name=eval_policy
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


export sif_image=/scratch/ppacaud/singularity_images/nvcuda_v2.sif
export python_bin=$HOME/miniconda3/envs/gembench/bin/python
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

expr_dir=data/experiments/gembench/3dlotus/v1
coarse_model_dir=data/experiments/gembench/3dlotus/v1_coarse

ckpt_step_coarse=150000

# validation

#for ckpt_step in 150000 140000 130000 80000 70000 60000 50000
#do
#singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
#    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_simple_policy_server_coarse_to_fine.py \
#    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 4 \
#    --taskvar_file assets/taskvars_train.json \
#    --seed 100 --num_demos 20 \
#    --microstep_data_dir data/gembench/val_dataset/microsteps/seed100 \
#    --coarse_model_dir ${coarse_model_dir} --ckpt_step_coarse ${ckpt_step_coarse}
#done

ckpt_step=140000
# test
for seed in {200..600..100}
do
for split in train test_l2 test_l3 test_l4
do
singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_simple_policy_server_coarse_to_fine.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 4 \
    --taskvar_file assets/taskvars_${split}.json \
    --seed ${seed} --num_demos 20 \
    --microstep_data_dir data/gembench/test_dataset/microsteps/seed${seed} \
    --coarse_model_dir ${coarse_model_dir} --ckpt_step_coarse ${ckpt_step_coarse}
done
done

