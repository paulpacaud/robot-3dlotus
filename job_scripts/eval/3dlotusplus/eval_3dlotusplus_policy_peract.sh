#!/bin/bash
#SBATCH --job-name=eval_motion_planner
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

expr_dir=data/experiments/peract/3dlotusplus/v1

# validation: with groundtruth task planner, and groundtruth object grounding
#for ckpt_step in 140000 130000 120000 110000 100000 90000 80000
#do
#singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
#    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_robot_pipeline_server_peract_gt.py \
#    --full_gt \
#    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline_gt.yaml \
#    --mp_expr_dir ${expr_dir} \
#    --mp_ckpt_step ${ckpt_step} \
#    --num_workers 4 \
#    --taskvar_file assets/taskvars_val_peract_wo_insert_peg.json \
#    --gt_og_label_file assets/taskvars_target_label_zrange_peract.json \
#    --gt_plan_file prompts/rlbench/in_context_examples_val_peract.txt \
#    --seed 100 \
#    --microstep_data_dir data/peract/val_dataset/microsteps/seed100 \
#    --pc_label_type fine --run_action_step 1
#done

# test: with groundtruth task planner and groundtruth object grounding
#ckpt_step=140000
#seed=200
#singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
#    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_robot_pipeline_server_peract_gt.py \
#    --full_gt \
#    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline_gt.yaml \
#    --mp_expr_dir ${expr_dir} \
#    --mp_ckpt_step ${ckpt_step} \
#    --num_workers 4 \
#    --taskvar_file assets/taskvars_test_peract.json \
#    --gt_og_label_file assets/taskvars_target_label_zrange_peract.json \
#    --gt_plan_file prompts/rlbench/in_context_examples_test_peract.txt \
#    --seed ${seed} \
#    --microstep_data_dir data/peract/test_dataset/microsteps/seed${seed} \
#    --pc_label_type fine --run_action_step 1
#    --record_video --video_dir ${expr_dir}/videos_llm_gt-og_gt/${split}/seed${seed} \
#    --not_include_robot_cameras  --video_rotate_cam \

# test: with groundtruth task planner and automatic object grounding
ckpt_step=140000
seed=200
singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_robot_pipeline_server_peract.py \
    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline.yaml \
    --mp_expr_dir ${expr_dir} \
    --mp_ckpt_step ${ckpt_step} \
    --num_workers 4 \
    --taskvar_file assets/taskvars_test_peract.json \
    --gt_plan_file prompts/rlbench/in_context_examples_test_peract.txt \
    --seed ${seed} --num_demos 20 \
    --microstep_data_dir data/peract/test_dataset/microsteps/seed${seed} \
    --prompt_dir prompts/rlbench/peract --asset_dir assets/peract \
    --pc_label_type fine --run_action_step 5

# test: full automatic
#seed=200
#ckpt_step=150000
#singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
#    xvfb-run -a ${python_bin} genrobo3d/evaluation/eval_robot_pipeline_server_peract.py \
#    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline.yaml \
#    --mp_expr_dir ${expr_dir} \
#    --mp_ckpt_step ${ckpt_step} \
#    --num_workers 4 \
#    --taskvar_file assets/taskvars_test_peract.json \
#    --seed ${seed} \
#    --microstep_data_dir data/peract/test_dataset/microsteps/seed${seed} \
#    --pc_label_type fine --run_action_step 5 \
#    --gt_plan_file prompts/rlbench/in_context_examples_test_peract.txt \
#    --prompt_dir prompts/rlbench/peract --asset_dir assets/peract \
#    --no_gt_llm --llm_master_port 15322
