import torch

model_name = "v1_voxel0.5cm_refine_zoom0.4_noise0.02"
ckpt_step = 100
model_checkpoint_file = f"data/experiments/gembench/3dlotus/{model_name}/ckpts/model_step_{ckpt_step}.pt"
# model_checkpoint_file = f"data/experiments/gembench/paul_test/ckpts/model_step_{ckpt_step}.pt"
checkpoint = torch.load(model_checkpoint_file, map_location=lambda storage, loc: storage)

print(f"All keys in checkpoint ({len(checkpoint.keys())} total):")
for k in checkpoint.keys():
    print(f"  {k}: {checkpoint[k].shape if isinstance(checkpoint[k], torch.Tensor) else type(checkpoint[k])}")

# Look specifically for embedding-related keys
print("\nEmbedding-related keys:")
for k in checkpoint.keys():
    if any(x in k for x in ['embedding', 'coarse_pred', 'pose']):
        print(f"  {k}: {checkpoint[k].shape if isinstance(checkpoint[k], torch.Tensor) else type(checkpoint[k])}")

from genrobo3d.configs.default import get_config
from genrobo3d.models.simple_policy_ptv3 import SimplePolicyPTV3CA
config = get_config('genrobo3d/configs/rlbench/simple_policy_ptv3.yaml')
model = SimplePolicyPTV3CA(config.MODEL).cuda()

print("Loaded model.state_dict")
for k in model.state_dict().keys():
    if any(x in k for x in ['embedding', 'coarse_pred', 'pose']):
        print(f"  {k}")
# model = SimplePolicyPTV3CA(config.MODEL).cuda()