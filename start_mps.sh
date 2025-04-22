#!/bin/bash

# Usage: ./start_mps.sh 0 2 3
# Starts an MPS daemon on each GPU ID provided

for GPU_ID in "$@"; do
  echo "Starting MPS for GPU $GPU_ID..."

  PIPE_DIR="/tmp/nvidia-mps-${GPU_ID}"
  LOG_DIR="/tmp/nvidia-log-${GPU_ID}"

  mkdir -p "$PIPE_DIR" "$LOG_DIR"
  chmod 777 "$PIPE_DIR" "$LOG_DIR"

  CUDA_VISIBLE_DEVICES=$GPU_ID \
  CUDA_MPS_PIPE_DIRECTORY=$PIPE_DIR \
  CUDA_MPS_LOG_DIRECTORY=$LOG_DIR \
  nvidia-cuda-mps-control -d
done
