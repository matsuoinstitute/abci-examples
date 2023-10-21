#!/bin/bash
#$ -l rt_G.small=1
#$ -j y
#$ -N test_model
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

deepspeed src/test_model_base.py
deepspeed src/test_model_peft.py