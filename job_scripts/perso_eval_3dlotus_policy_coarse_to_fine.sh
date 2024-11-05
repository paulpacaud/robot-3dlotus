expr_dir=data/experiments/gembench/3dlotus/v1
coarse_model_dir=data/experiments/gembench/3dlotus/v1_coarse

ckpt_step_coarse=150000
ckpt_step=150000

python genrobo3d/evaluation/eval_simple_policy_server_coarse_to_fine.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 4 \
    --taskvar_file assets/taskvars_train.json \
    --seed 100 --num_demos 20 \
    --microstep_data_dir data/gembench/val_dataset/microsteps/seed100 \
    --coarse_model_dir ${coarse_model_dir} --ckpt_step_coarse ${ckpt_step_coarse}
