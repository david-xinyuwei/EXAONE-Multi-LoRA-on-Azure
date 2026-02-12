#!/bin/bash
# Start vLLM server with dynamic LoRA adapter support.
# Unlike static mode (--lora-modules), adapters are loaded/unloaded at runtime
# via /v1/load_lora_adapter and /v1/unload_lora_adapter endpoints.
#
# Author: Hyeonsang Jeon
#
# Usage:
#   chmod +x start_dynamic_serving.sh
#   ./start_dynamic_serving.sh
#
# Then load adapters dynamically:
#   curl -X POST http://localhost:8080/v1/load_lora_adapter \
#     -H "Content-Type: application/json" \
#     -d '{"lora_name": "medical", "lora_path": "/data/lora-adapters/medical"}' 

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/EXAONE-3.5-2.4B-Instruct}"
PORT="${PORT:-8080}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
MAX_LORAS="${MAX_LORAS:-4}"          # Max adapters in GPU memory
MAX_CPU_LORAS="${MAX_CPU_LORAS:-8}"  # Max adapters in CPU memory (tier 2 cache)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

echo "============================================"
echo "  vLLM Dynamic LoRA Serving"
echo "============================================"
echo "  Model:          ${MODEL_PATH}"
echo "  Port:           ${PORT}"
echo "  Max LoRA Rank:  ${MAX_LORA_RANK}"
echo "  Max GPU LoRAs:  ${MAX_LORAS}"
echo "  Max CPU LoRAs:  ${MAX_CPU_LORAS}"
echo "  GPU Mem Util:   ${GPU_MEM_UTIL}"
echo "  Dynamic Swap:   ENABLED"
echo "============================================"

VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --max-loras "${MAX_LORAS}" \
  --max-cpu-loras "${MAX_CPU_LORAS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --dtype bfloat16
