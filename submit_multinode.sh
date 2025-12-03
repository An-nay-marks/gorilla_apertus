#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=bfcl_eval_apertus
#SBATCH --partition=normal
#SBATCH --time=00:05:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output logs/slurm-%x-%j.out
#SBATCH --error logs/slurm-%x-%j.err
#SBATCH --environment=/iopsstor/scratch/cscs/amarx/envs/bfcl-01.toml

set -x

# 1. Generate Hostfile for DeepSpeed
# This tells DeepSpeed exactly which nodes SLURM gave you
export HOSTFILE=hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > $HOSTFILE
echo "Hostfile content:"
cat $HOSTFILE

ACCEL_PROCS=$(( $SLURM_NNODES * $SLURM_GPUS_PER_NODE ))

MAIN_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MAIN_PORT=12802

# REPLACE WITH YOUR BASE PATH, EASIER THAN CORRECTING ALL LINES 
# cannot use your values because permission denied 
#BASEPATH = "/users/sinievdben/scratch/sinievdben/project/project_git"
export BASE_PATH="/iopsstor/scratch/cscs/amarx/apertus-finetuning-project"

export HF_HOME="$BASE_PATH/cache/huggingface"
export HF_DATASETS_CACHE="$BASE_PATH/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$BASE_PATH/cache/huggingface/transformers"

# Create these directories if they don't exist
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE
mkdir -p logs  # Ensure log directory exists for SLURM output

# export TORCH_EXTENSIONS_DIR="/iopsstor/scratch/cscs/tstaehle/torch_extensions"
# mkdir -p $TORCH_EXTENSIONS_DIR

# (Optional but recommended) Redirect Triton cache if you use Flash Attention
export TRITON_CACHE_DIR="$BASE_PATH/.triton/cache"
mkdir -p $TRITON_CACHE_DIR

export WANDB_API_KEY="538e84b2e22079e598e6c77a14bed6558753c0c0" # your previous submission contained your api key
export WANDB_PROJECT="bfcl-eval-apertus"
export WANDB_ENTITY="apertus"
export WANDB_NAME="apertus-finetune-bfcl-eval-$(date +%Y%m%d-%H%M%S)"
export WANDB_NOTES="Berkley Function Call Leaderboard Evaluation on Apertus"

export BFCL_PROJECT_ROOT="$BASE_PATH/gorilla_apertus"

# save your trained model checkpoint to wandb
#export WANDB_LOG_MODEL="true"

# turn off watch to log faster
export WANDB_WATCH="false"
cd "$BASE_PATH/gorilla_apertus/berkeley-function-call-leaderboard/"

# python -m bfcl_eval._llm_response_generation --model ApertusFC --test-category simple --result-dir results_apertus --num-threads 1 --include-input-log
bfcl generate --model ApertusFC --test-category simple_python --result-dir results_apertus --num-threads 1 --include-input-log