#!/bin/bash
#
# Training script for small Rust code generation model
#
# Configuration:
# - Model depth: 8 (small model with ~8 layers)
# - Model dimension: 512 (depth * aspect_ratio = 8 * 64)
# - Sequence length: 1024 (suitable for code)
# - Batch size: 8 (memory efficient)
# - Learning rates: reduced for small model stability
#
# Usage:
#   ./runs/rust_train.sh
#
# Or with custom settings:
#   DEPTH=8 NPROC_PER_NODE=1 ./runs/rust_train.sh
#

set -e

# Environment setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR

# Create rust data directory if it doesn't exist
RUST_DATA_DIR="$NANOCHAT_BASE_DIR/rust_data"
mkdir -p $RUST_DATA_DIR

# Configuration
SERIES_NAME="${SERIES_NAME:-rust_small}"
DEPTH="${DEPTH:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
WANDB_RUN="${WANDB_RUN:-rust_train_d${DEPTH}}"

# Model tag for checkpoint naming
MODEL_TAG="rust_d${DEPTH}"

# Setup (skip with SKIP_SETUP=1)
if [ -z "$SKIP_SETUP" ]; then
    echo "Setting up environment..."
    
    # uv
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate

    # Ensure tokenizer exists (use existing or train if needed)
    if [ ! -d "$NANOCHAT_BASE_DIR/tokenizer" ]; then
        echo "Training tokenizer..."
        # Download some data for tokenizer training if not available
        if [ ! -d "$NANOCHAT_BASE_DIR/base_data_climbmix" ]; then
            python -m nanochat.dataset -n 10
        fi
        python -m scripts.tok_train --max-chars=200000000 --vocab-size=32768
    fi

    # Prepare Rust data - copy to expected location or symlink
    echo "Preparing Rust data..."
    RUST_DATA_SOURCE="data/rust_code_train.parquet"
    
    if [ -f "$RUST_DATA_SOURCE" ]; then
        # Copy the rust data to the expected location
        # The dataloader expects parquet files in base_data_climbmix
        # We'll rename it to look like a valid shard
        cp "$RUST_DATA_SOURCE" "$RUST_DATA_DIR/shard_00000.parquet"
        
        # Also create a symlink in the main data directory if needed
        mkdir -p "$NANOCHAT_BASE_DIR/base_data_climbmix"
        cp "$RUST_DATA_SOURCE" "$NANOCHAT_BASE_DIR/base_data_climbmix/shard_00000.parquet"
        
        # Create validation split (use a portion as validation)
        # For simplicity, we'll use the same file but the dataloader will use last file as val
        echo "Using $RUST_DATA_SOURCE for training"
    else
        echo "ERROR: Rust data file not found at $RUST_DATA_SOURCE"
        echo "Please run: python scripts/prepare_rust_data.py"
        exit 1
    fi
else
    echo "Skipping setup (SKIP_SETUP=1)"
    source .venv/bin/activate
fi

echo "=============================================="
echo "Rust Code Generation Model Training"
echo "=============================================="
echo "Model: depth=$DEPTH, seq_len=$MAX_SEQ_LEN"
echo "Batch size: $DEVICE_BATCH_SIZE"
echo "Data: $RUST_DATA_SOURCE"
echo "Checkpoint: $MODEL_TAG"
echo "=============================================="

# Run training
# Using torchrun for distributed training
# Key parameters:
# --depth: Number of transformer layers (8 for small model)
# --max-seq-len: Sequence length (1024 for code)
# --device-batch-size: Batch size per device
# --eval-every: Evaluate every N steps (250)
# --save-every: Save checkpoint every N steps (500)
# --run: wandb run name
# --model-tag: Checkpoint directory name
# --target-param-data-ratio: Tokens to params ratio (Chinchilla-optimal ~10.5)
# --embedding-lr, --matrix-lr: Learning rates (reduced for small model)
# --warmup-steps: Fewer warmup steps for smaller training run

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    --depth=$DEPTH \
    --max-seq-len=$MAX_SEQ_LEN \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --eval-every=250 \
    --save-every=500 \
    --target-param-data-ratio=10.5 \
    --core-metric-every=999999 \
    --core-metric-max-per-task=-1 \
    --sample-every=1000 \
    --embedding-lr=0.15 \
    --unembedding-lr=0.004 \
    --matrix-lr=0.01 \
    --warmup-steps=20 \
    --window-pattern=L \
    2>&1 | tee "$NANOCHAT_BASE_DIR/${MODEL_TAG}_train.log"

echo "=============================================="
echo "Training complete!"
echo "Checkpoints saved to: $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
echo "Log: $NANOCHAT_BASE_DIR/${MODEL_TAG}_train.log"
echo "=============================================="