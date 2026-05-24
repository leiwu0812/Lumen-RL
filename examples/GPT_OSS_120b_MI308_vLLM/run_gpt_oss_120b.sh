#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# GPT-OSS-120B Eagle3 Draft Model Distillation (vLLM + Mooncake TCP) — MI308
#
# Off-policy speculative decoding draft model training using teacher hidden
# states. vLLM 0.11 with extract_hidden_states mode provides teacher inference
# on ROCm/AMD MI308 GPUs; Mooncake TCP transfers hidden states to training GPUs
# running LumenRL's aiter-patched FSDP2.
#
# GPU split strategy (8x MI308):
#   GPUs 0-3: torchrun FSDP2 draft model training (BF16, LumenRL + aiter)
#   GPUs 4-7: vLLM teacher inference (TP=4, native MXFP4 MoE)
#
# Usage:
#   bash examples/GPT_OSS_120b_MI308_vLLM/run_gpt_oss_120b.sh
#   bash examples/GPT_OSS_120b_MI308_vLLM/run_gpt_oss_120b.sh --smoke-test
#   MODEL_PATH=/path/to/model bash examples/GPT_OSS_120b_MI308_vLLM/run_gpt_oss_120b.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXP_NAME="gpt-oss-120b-eagle3-vllm-mi308"
OUTPUT_DIR="${REPO_ROOT}/output/GPT_OSS_120b_SDDD/LumenRL"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"

# GPU split (managed by VllmTeacherEngine subprocess for the inference half):
#   0-3 training, 4-7 inference (TP=4).  Override via env vars if needed.
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"

# Model / data paths (default to /dev/shm)
MODEL_PATH="${MODEL_PATH:-/dev/shm/gpt-oss-120b}"
if [ "${SMOKE_TEST}" = true ]; then
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/gpt_oss_120b_smoke_test_vllm}"
else
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/gpt_oss_120b_eagle3_vllm}"
fi

# Config selection
if [ "${SMOKE_TEST}" = true ]; then
    CONFIG="${SCRIPT_DIR}/configs/smoke_test.yaml"
    echo ">>> SMOKE TEST: 5-step Eagle3 validation (vLLM+Mooncake TCP, gpt-oss-120b)"
else
    CONFIG="${SCRIPT_DIR}/configs/opd_gpt_oss_120b.yaml"
fi

# Environment
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LUMENRL_LOG_LEVEL=INFO
export NCCL_TIMEOUT=7200

mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  GPT-OSS-120B Eagle3 Draft Model Distillation (vLLM) — MI308"
echo "  Teacher:     ${MODEL_PATH} (vLLM 0.11, GPUs 4-7, TP=4, native MXFP4)"
echo "  Draft:       Eagle3 (from scratch, 1 Transformer block, BF16)"
echo "  Train GPUs:  ${TRAIN_GPUS} (${NUM_TRAIN_GPUS} GPUs, FSDP2+aiter)"
echo "  Transfer:    Mooncake TCP"
echo "  Config:      ${CONFIG}"
echo "  Output:      ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# Build overrides
OVERRIDES=(
    "policy.model_name=${MODEL_PATH}"
    "algorithm.teacher.model_name=${MODEL_PATH}"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
)

# Launch training on GPUs 0-3 (VllmTeacherEngine subprocess manages GPUs 4-7 internally)
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" \
        -m lumenrl.trainer.main \
        --config "${CONFIG}" \
        ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
        2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> GPT-OSS-120B Eagle3 distillation (vLLM, MI308) completed successfully."
    echo ">>> Logs saved to: ${LOG_FILE}"
else
    echo ">>> GPT-OSS-120B Eagle3 distillation (vLLM, MI308) failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
