#!/bin/bash

set -euo pipefail

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# Run as:
# bash dev/cpu_demo_run.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# This is also why I hide this script away in dev/

# all the setup stuff
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

# Prefer uv if available (fast, handles torch index selection), otherwise fall back to python3 + venv + pip.
if command -v uv &> /dev/null; then
    USE_UV=1
    [ -d ".venv" ] || uv venv
    uv sync --extra cpu
    source .venv/bin/activate
else
    USE_UV=0
    if command -v python &> /dev/null; then
        PYTHON_BIN="python"
    elif command -v python3 &> /dev/null; then
        PYTHON_BIN="python3"
    else
        echo "Error: neither 'python' nor 'python3' is available on PATH." 1>&2
        echo "Install Python 3.10+ and try again." 1>&2
        exit 1
    fi

    [ -d ".venv" ] || "$PYTHON_BIN" -m venv .venv
    source .venv/bin/activate

    # Install runtime deps (without installing this project as a package).
    python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade
    python -m pip install -U pip
    python -m pip install -U \
        maturin \
        datasets \
        fastapi \
        files-to-prompt \
        psutil \
        regex \
        setuptools \
        tiktoken \
        tokenizers \
        torch \
        uvicorn \
        wandb
fi
if [ -z "${WANDB_RUN:-}" ]; then
    export WANDB_RUN=dummy
fi
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
if [ "$USE_UV" -eq 1 ]; then
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
else
    python -m maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# wipe the report
python -m nanochat.report reset

# Download identity conversation dataset if missing (used in midtraining and SFT).
IDENTITY_CONV_PATH="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_CONV_PATH" ]; then
    curl -L -o "$IDENTITY_CONV_PATH" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# train tokenizer on ~1B characters
python -m nanochat.dataset -n 4
python -m scripts.tok_train --max_chars=1000000000
python -m scripts.tok_eval

# train a very small 4 layer model
# For macOS (especially with MPS), keep this tiny to avoid out-of-memory during eval.
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --eval_every=10 \
    --eval_tokens=512 \
    --core_metric_every=-1 \
    --sample_every=1000000000 \
    --num_iterations=20

# Run heavier eval steps on CPU for stability.
python -m scripts.base_loss --device_type=cpu --device_batch_size=1 --split_tokens=2048
python -m scripts.base_eval --device-type=cpu --max-per-task=16

# midtraining
python -m scripts.mid_train \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --eval_every=50 \
    --eval_tokens=512 \
    --total_batch_size=512 \
    --num_iterations=50
# eval results will be terrible, this is just to execute the code paths.
# note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
python -m scripts.chat_eval --source=mid --device-type=cpu --max-new-tokens=128 --max-problems=20

# SFT
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --num_iterations=20 \
    --eval_steps=4 \
    --eval_metrics_max_problems=16

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate
