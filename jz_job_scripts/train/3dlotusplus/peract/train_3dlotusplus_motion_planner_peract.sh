#!/bin/bash
#SBATCH --job-name=trMP
#SBATCH -A hjx@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
##SBATCH --mem=40G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out

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

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_TASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

ulimit -n 2048

benchmark=peract
label_type=coarse
output_dir=data/experiments/${benchmark}/3dlotusplus/v2_${label_type}

rot_type=euler_disc
npoints=4096
pos_bin_size=15
max_traj_len=5

if [ "$benchmark" = "gembench" ]; then
    script_path="genrobo3d/train/train_motion_planner.py"
elif [ "$benchmark" = "peract" ]; then
    script_path="genrobo3d/train/train_motion_planner_peract.py"
else
    echo "Error: Unknown benchmark ${benchmark}"
    exit 1
fi

srun python ${script_path} \
    --exp-config genrobo3d/configs/rlbench/motion_planner_ptv3.yaml \
    output_dir ${output_dir} \
    TRAIN.num_epochs null \
    TRAIN.num_train_steps 180000 \
    TRAIN.log_steps 1000 \
    TRAIN.save_steps 10000 \
    TRAIN.val_steps 10000 \
    TRAIN.train_batch_size 8 \
    TRAIN.val_batch_size 8 \
    VAL_DATASET.use_val False \
    TRAIN_DATASET.rm_robot box_keep_gripper \
    VAL_DATASET.rm_robot box_keep_gripper \
    TRAIN_DATASET.num_points ${npoints} \
    VAL_DATASET.num_points ${npoints} \
    TRAIN_DATASET.all_step_in_batch True \
    VAL_DATASET.all_step_in_batch True \
    TRAIN_DATASET.instr_embed_type all \
    VAL_DATASET.instr_embed_type all \
    TRAIN_DATASET.xyz_shift center \
    VAL_DATASET.xyz_shift center \
    TRAIN_DATASET.xyz_norm False \
    VAL_DATASET.xyz_norm False \
    TRAIN_DATASET.rot_type ${rot_type} \
    VAL_DATASET.rot_type ${rot_type} \
    TRAIN_DATASET.use_height True \
    VAL_DATASET.use_height True \
    TRAIN_DATASET.augment_pc True \
    VAL_DATASET.augment_pc False \
    TRAIN_DATASET.aug_max_rot 45 \
    TRAIN_DATASET.rm_pc_outliers False \
    VAL_DATASET.rm_pc_outliers False \
    TRAIN_DATASET.max_traj_len ${max_traj_len} \
    VAL_DATASET.max_traj_len ${max_traj_len} \
    TRAIN_DATASET.pc_label_type ${label_type} \
    VAL_DATASET.pc_label_type ${label_type} \
    TRAIN_DATASET.pc_label_augment 0.0 \
    VAL_DATASET.pc_label_augment 0.0 \
    TRAIN_DATASET.pc_midstep_augment True \
    VAL_DATASET.pc_midstep_augment True \
    TRAIN_DATASET.data_dir data/${benchmark}/train_dataset/motion_keysteps_bbox_pcd/seed0/voxel1cm \
    TRAIN_DATASET.taskvar_file assets/${benchmark}/taskvars_train_${benchmark}.json \
    VAL_DATASET.taskvar_file assets/${benchmark}/taskvars_train_${benchmark}.json \
    TRAIN_DATASET.gt_act_obj_label_file assets/${benchmark}/taskvars_target_label_zrange_${benchmark}.json \
    VAL_DATASET.gt_act_obj_label_file assets/${benchmark}/taskvars_target_label_zrange_${benchmark}.json \
    TRAIN_DATASET.instr_include_objects False \
    VAL_DATASET.instr_include_objects False \
    TRAIN_DATASET.action_embed_file data/${benchmark}/train_dataset/motion_keysteps_bbox_pcd/action_embeds_clip.npy \
    VAL_DATASET.action_embed_file data/${benchmark}/train_dataset/motion_keysteps_bbox_pcd/action_embeds_clip.npy \
    TRAIN_DATASET.use_color False \
    VAL_DATASET.use_color False \
    MODEL.ptv3_config.drop_path 0.0 \
    MODEL.ptv3_config.attn_drop 0.1 \
    MODEL.ptv3_config.proj_drop 0.1 \
    MODEL.action_config.dropout 0.2 \
    MODEL.action_config.voxel_size 0.01 \
    MODEL.action_config.reduce max \
    MODEL.action_config.dim_actions 7 \
    MODEL.action_config.rot_pred_type ${rot_type} \
    MODEL.action_config.pos_pred_type heatmap_disc \
    MODEL.action_config.pos_heatmap_temp 0.1 \
    MODEL.ptv3_config.in_channels 4 \
    MODEL.ptv3_config.pdnorm_only_decoder False \
    MODEL.ptv3_config.qk_norm True \
    MODEL.ptv3_config.scaled_cosine_attn False \
    MODEL.ptv3_config.enable_flash True \
    MODEL.action_config.max_steps 30 \
    MODEL.ptv3_config.enc_depths "[1, 1, 1, 1, 1]" \
    MODEL.ptv3_config.dec_depths "[1, 1, 1, 1]" \
    MODEL.ptv3_config.enc_channels "[64, 128, 256, 512, 768]" \
    MODEL.ptv3_config.dec_channels "[128, 128, 256, 512]" \
    MODEL.loss_config.pos_weight 1 \
    MODEL.loss_config.rot_weight 1 \
    TRAIN_DATASET.pos_type disc \
    VAL_DATASET.pos_type disc \
    TRAIN_DATASET.pos_heatmap_type dist \
    VAL_DATASET.pos_heatmap_type dist \
    MODEL.action_config.max_traj_len ${max_traj_len} \
    TRAIN_DATASET.pos_bins ${pos_bin_size} \
    VAL_DATASET.pos_bins ${pos_bin_size} \
    MODEL.action_config.pos_bins ${pos_bin_size} \
    TRAIN_DATASET.pos_heatmap_no_robot True \
    VAL_DATASET.pos_heatmap_no_robot True \
    MODEL.action_config.txt_reduce attn \
    MODEL.action_config.use_ee_pose False \
    MODEL.model_class MotionPlannerPTV3CA \
    MODEL.ptv3_config.pdnorm_bn False \
    MODEL.ptv3_config.pdnorm_ln False \
    MODEL.ptv3_config.pdnorm_adaptive False
    

