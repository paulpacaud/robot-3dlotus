# Installation Instructions

1. Install general python packages
```bash
conda create -n gembench python==3.10

conda activate gembench


#On CLEPS or some specific HPC clusters, you may need: `module load gnu12/12.2.0` 
conda install nvidia/label/cuda-12.1.0::cuda

# on HPC Cluster (like JZ), replace $HOME by $WORK
export CUDA_HOME=$HOME/.conda/envs/gembench # change here the path to your conda environment
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# On Jean-Zay or some specific HPC clusters, you may need: `module load gcc/11.3.1` for gnu-c++ errors

# some people report issues with conda-forge when installing torch related packages, due to the crypt.h file missing. Refer to this thread for help: https://github.com/stanford-futuredata/ColBERT/issues/309
# takeaway: if problem finding crypt.h file, it is likely because you need to cp it from /usr/include to $HOME/.conda/envs/gembench/include/python3.10

### Everywhere
pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip install --no-cache-dir -r requirements.txt
# if issues with torch_scatter, try: pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html


# install genrobo3d
pip install --no-cache-dir -e .
```

2. Install RLBench
```bash
mkdir dependencies
cd dependencies
```

Download CoppeliaSim (see instructions [here](https://github.com/stepjam/PyRep?tab=readme-ov-file#install))
```bash
# change the version if necessary
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

Add the following to your ~/.bashrc file:
```bash
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Install Pyrep and RLBench
```bash
git clone https://github.com/cshizhe/PyRep.git
cd PyRep
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir .
cd ..

# Our modified version of RLBench to support new tasks in GemBench
git clone https://github.com/rjgpinel/RLBench
cd RLBench
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir .
cd ../..
```

3. Install model dependenciessa

```bash
cd dependencies

# please ensure to set CUDA_HOME beforehand as specified in the export const of the section 1
# This covers V100 (7.0), A100 (8.0), and H100 (9.0)
export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
export FORCE_CUDA=1
git clone https://github.com/cshizhe/chamferdist.git
cd chamferdist
python setup.py install
cd ..

# you may need to set export MAX_JOBS=2 or 1 before running the following commands because of the limited resources of your cluster
git clone https://github.com/cshizhe/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
python setup.py install
cd ../..

# llama3: needed for 3D-LOTUS++
git clone https://github.com/cshizhe/llama3.git
cd llama3
pip install --no-cache-dir -e .
# if you downloaded llama3.1, you need to modify the model.py file as follows:
cd llama
nano model.py; add use_scaled_rope: bool = True at line 33 #https://github.com/meta-llama/llama3/issues/291
cd ../..

# Choose where you want to store the pretrained models LLM, Encoders.. We choose data/pretrained here:
cd data/pretrained
# Download llama3-8B model following instructions here: https://github.com/cshizhe/llama3?tab=readme-ov-file#download, and modify the configuration path in genrobo3d/configs/rlbench/robot_pipeline.yaml
# For Llama, you may need to change the download folder of the model to point to a large memory folder by changing 
export LLAMA_STACK_CONFIG_DIR=...

# You will also need to download pretrained models
# Hugging Face cache models on a cache directory, you can change the cache directory by setting the HF_HOME environment variable, in the terminal or in your bashrc:
export HF_HOME=$WORK/.cache/huggingface
mkdir -p $HF_HOME

module load miniforge/24.9.0
conda activate gembench
python
>>> from transformers import CLIPModel, AutoTokenizer, CLIPProcessor

>>> model_name='openai/clip-vit-base-patch32'
>>> model = CLIPModel.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
>>> processor = CLIPProcessor.from_pretrained(model_name)

>>> model.save_pretrained("./clip-vit-base-patch32")
>>> tokenizer.save_pretrained("./clip-vit-base-patch32")
>>> processor.save_pretrained("./clip-vit-base-patch32")

>>> from transformers import AutoModel
>>> bert_model='sentence-transformers/all-MiniLM-L6-v2'
>>> bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
>>> bert_model = AutoModel.from_pretrained(bert_model)

>>> bert_tokenizer.save_pretrained("./all-MiniLM-L6-v2")
>>> bert_model.save_pretrained("./all-MiniLM-L6-v2")

>>> from transformers import Owlv2Processor, Owlv2ForObjectDetection
>>> owlv2_processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
>>> owlv2_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")

>>> owlv2_processor.save_pretrained("./owlv2-large-patch14-ensemble")
>>> owlv2_model.save_pretrained("./owlv2-large-patch14-ensemble")

>>> from transformers import SamProcessor, SamModel
### TO REDO
>>> sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
>>> sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")

>>> sam_processor.save_pretrained("./sam-vit-huge")
>>> sam_model.save_pretrained("./sam-vit-huge")

# If the save_pretrained command is killed, it may stem from a out of memory issue, check it using: dmesg -T| grep -E -i -B100 'killed process'
# If it happens, you can try downloading it manually in python interactive mode:
import torch
from huggingface_hub import hf_hub_download
import os
import shutil

# Create directory
os.makedirs("sam-vit-huge", exist_ok=True)

# Download configuration files
files = ["config.json", "preprocessor_config.json"]
for file in files:
    hf_hub_download(
        repo_id="facebook/sam-vit-huge",
        filename=file,
        local_dir="sam-vit-huge",
        local_dir_use_symlinks=False
    )

# Download the model file separately
model_file = hf_hub_download(
    repo_id="facebook/sam-vit-huge",
    filename="model.safetensors",
    local_dir="sam-vit-huge",
    local_dir_use_symlinks=False,
    resume_download=True  # This will resume if download is interrupted
)

# Verify the files
print("Downloaded files:", os.listdir("sam-vit-huge"))

```

4. Running headless

If you have sudo priviledge on the headless machine, you could follow [this instruction](https://github.com/rjgpinel/RLBench?tab=readme-ov-file#running-headless) to run RLBench.

Otherwise, you can use [singularity](https://apptainer.org/docs/user/1.3/index.html) or [docker](https://docs.docker.com/) to run RLBench in headless machines without sudo privilege.
The [XVFB](https://manpages.ubuntu.com/manpages/xenial/man1/xvfb-run.1.html) should be installed in the virtual image in order to have a virtual screen.

A pre-built singularity image can be downloaded [here](https://www.dropbox.com/scl/fi/wnf27yd4pkeywjk2y3wd4/nvcuda_v2.sif?rlkey=7lpni7d9b6dwjj4wehldq8037&st=5steya0b&dl=0).
Here are some simple commands to run singularity:
```bash
export SINGULARITY_IMAGE_PATH=`YOUR PATH TO THE SINGULARITY IMAGE`
export python_bin=$HOME/miniconda3/envs/gembench/bin/python

# interactive mode
singularity shell --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH

# run script
singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH xvfb-run -a ${python_bin} ...
```

5. Adapt the codebase to your environment

To adapt the codebase to your environment, you may need to modify the following:
- replace everywhere $HOME/codes/robot-3dlotus with your path to robot-3dlotus folder
- replace everywhere the sif_image path to your path to the singularity image nvcuda_v2.sif
