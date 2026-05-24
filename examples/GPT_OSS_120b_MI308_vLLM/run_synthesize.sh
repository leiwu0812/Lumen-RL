#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# GPT-OSS-120B SpecForge-style data synthesis — Docker launcher.
#
# Runs synthesize_specforge_data.py inside the same container as training,
# using all 8 MI308 GPUs (TP=8) since the training job is not co-tenant
# during synthesis.
#
# Output: /dev/shm/gpt_oss_120b_synth/{ultrachat_200k_synth,magpie_300k_synth}.jsonl
#
# NOTE: /dev/shm is tmpfs — files are wiped on host reboot. Copy to durable
# storage when the run completes.
#
# Usage:
#   bash examples/GPT_OSS_120b_MI308_vLLM/run_synthesize.sh
#   bash examples/GPT_OSS_120b_MI308_vLLM/run_synthesize.sh --smoke-test
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

DOCKER_IMAGE="${DOCKER_IMAGE:-lumenrl-vllm-mi308:latest}"
CONTAINER_NAME="gpt_oss_120b_specforge_synth"
LUMENRL_DIR="/home/leiwu/Lumen-RL"
MODEL_PATH="${MODEL_PATH:-/dev/shm/gpt-oss-120b}"
OUTPUT_DIR="/dev/shm/gpt_oss_120b_synth"

SYNTH_CMD="python -u examples/GPT_OSS_120b_MI308_vLLM/synthesize_specforge_data.py \
    --model ${MODEL_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --tp-size 8 \
    --max-tokens 2048 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --batch-prompts 4096 \
    --seed 42"

if [ "${SMOKE_TEST}" = true ]; then
    SYNTH_CMD="${SYNTH_CMD} --per-dataset-limit 100 --datasets ultrachat"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  GPT-OSS-120B SpecForge Synthesis — SMOKE TEST"
    echo "  100 prompts from ultrachat only, ~5-10 min"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "═══════════════════════════════════════════════════════════════"
    echo "  GPT-OSS-120B SpecForge Synthesis — FULL RUN"
    echo "  ~500K prompts (ultrachat ~207K + Magpie ~300K), ~30-50h"
    echo "  Resumable — safe to ctrl-C and rerun the same command."
    echo "═══════════════════════════════════════════════════════════════"
fi

RUN_CMD="mkdir -p ${OUTPUT_DIR} && ${SYNTH_CMD} 2>&1 | tee -a ${OUTPUT_DIR}/synth.log"

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    --shm-size 64G \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e PYTORCH_ROCM_ARCH=gfx942 \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e LUMENRL_LOG_LEVEL=INFO \
    -e NCCL_TIMEOUT=7200 \
    -e WANDB_MODE=disabled \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v /dev/shm:/dev/shm \
    -v "${LUMENRL_DIR}:/root/lumenrl" \
    -w /root/lumenrl \
    "${DOCKER_IMAGE}" \
    bash -c "${RUN_CMD}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Synthesis container exited cleanly."
    echo ">>> Outputs: ${OUTPUT_DIR}/"
    ls -lh "${OUTPUT_DIR}/" 2>/dev/null || true
else
    echo ">>> Synthesis container failed (exit ${EXIT_CODE}); rerun the same command to resume." >&2
fi
exit ${EXIT_CODE}
