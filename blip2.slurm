#!/bin/bash

#SBATCH -J BLIP2VGPT
#SBATCH -p gpu
#SBATCH -A c01949
#SBATCH -o blip2_outputs.txt
#SBATCH -e blip2_error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jkatama@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00

#Load any modules that your program needs

cd /N/u/jkatama/BigRed200/BLIP2/ICapGPT/

module load python/gpu

python -m venv .venv

source .venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install transformers datasets matplotlib 

srun ./gpt_training1.py 