# GPT-OSS-120B Eagle3 Draft Distillation (vLLM + Mooncake TCP) — MI308

Train Eagle3 speculative decoding draft model using OpenAI's `gpt-oss-120b` (117B-param MoE, 5.1B active) teacher hidden states on **8x MI308 GPUs** with vLLM inference and Mooncake TCP transfer.

- **GPUs 0-3**: torchrun FSDP2 -- Eagle3 draft model training (BF16, LumenRL + aiter)
- **GPUs 4-7**: vLLM -- gpt-oss-120b teacher (TP=4, native MXFP4 MoE)

## Architecture

```
Training GPUs (0-3)                    Inference GPUs (4-7)
LumenRL FSDP2 + aiter          <---   vLLM 0.19.1 (extract_hidden_states)
  Eagle3 draft model, BF16              TP=4, native MXFP4 MoE
       ^                                       |
  Mooncake TCP  <---------------------  hidden_states via
  EagleMooncakeStore                   MooncakeHiddenStatesConnector
```

## Docker image

**Reuses `lumenrl-vllm-mi308:latest`** built by the sibling Qwen3-8B example. No
separate Dockerfile is needed here because:

- vLLM 0.19.1 (in the existing image) already registers `GptOssForCausalLM` and
  loads MXFP4 weights natively.
- `vllm/model_executor/models/gpt_oss.py` already inherits from `EagleModelMixin`
  and implements `_maybe_add_hidden_state`, so the existing
  `examples/Qwen3_8B_SDDD_MI308_vLLM/docker/patches/vllm/v0.19.1/*.patch`
  cover the extract_hidden_states pipeline without any gpt-oss-specific addition.

If the image isn't built yet:
```bash
bash examples/Qwen3_8B_SDDD_MI308_vLLM/docker/build.sh
```

## Quick Start

### 1. Download model

```bash
huggingface-cli download openai/gpt-oss-120b --local-dir /dev/shm/gpt-oss-120b
```
(~196 GB MXFP4 distribution; large download.)

### 2. Smoke test (5 steps, synthetic prompts)

```bash
bash examples/GPT_OSS_120b_MI308_vLLM/run_docker.sh --smoke-test
# logs → output/GPT_OSS_120b_SDDD/LumenRL/
```

### 3. Full training (500 steps, NuminaMath-TIR)

```bash
bash examples/GPT_OSS_120b_MI308_vLLM/run_docker.sh
```

Override env vars as needed:
```bash
MODEL_PATH=/some/other/path \
CKPT_DIR=/some/checkpoint/dir \
DOCKER_IMAGE=lumenrl-vllm-mi308:my-tag \
HF_TOKEN=hf_xxx \
bash examples/GPT_OSS_120b_MI308_vLLM/run_docker.sh
```

## Model facts (gpt-oss-120b)

| | |
|---|---|
| Total params | 117B |
| Active params / token | 5.1B (MoE: 128 experts, 4 per token) |
| Layers | 36 |
| Hidden | 2880 |
| Heads / KV heads | 64 / 8 (GQA) |
| Vocab | 201088 (`o200k_harmony` tokenizer) |
| RoPE | θ=150000 + YaRN(factor=32, original_max=4096) |
| Native quantization | MXFP4 for MoE weights (~196 GB on disk) |
| Eagle3 aux layers (auto, N=36) | `[2, 18, 33]` |

## File structure

```
configs/
  opd_gpt_oss_120b.yaml    # 500-step training on AI-MO/NuminaMath-TIR
  smoke_test.yaml          # 5-step e2e validation, synthetic prompts
run_gpt_oss_120b.sh        # In-container entrypoint (torchrun + overrides)
run_docker.sh              # Host-side wrapper, launches container
```

No `docker/` subdir: the image is shared with `Qwen3_8B_SDDD_MI308_vLLM`.

## Environment variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | All 8 GPUs |
| `PYTORCH_ROCM_ARCH` | `gfx942` | MI308 GPU architecture |
| `HIP_FORCE_DEV_KERNARG` | `1` | Required for MI308 kernel args |
| `WANDB_MODE` | `disabled` | Disable W&B logging for smoke tests |
| `MODEL_PATH` | `/dev/shm/gpt-oss-120b` (override) | Teacher / policy model path |
| `DOCKER_IMAGE` | `lumenrl-vllm-mi308:latest` (override) | Container image tag |

## Config notes

- `algorithm.teacher.quantization: ""` — gpt-oss-120b ships natively as MXFP4
  (see HF config `quantization_config.quant_method=mxfp4`); vLLM loads it
  as-is from the checkpoint directory. We do **not** pass `quantization=mxfp4`
  to vLLM args, which would re-route through the ATOM online-quantization
  plugin (built for BF16-shipped models and trips a `list(None)` bug on
  non-quark configs).
- `policy.max_total_sequence_length: 16384` — gpt-oss native is 131K (YaRN
  factor 32) but Eagle3 draft training doesn't need full context; capping
  reduces mooncake hidden-state buffer pressure.
- `policy.learning_rate: 5e-5` — lower than Qwen3-8B's `1e-4`, matching Kimi
  K25 (larger model → smaller LR).
- AITER acceleration features (`lumen_norm`, `lumen_linear`, `hf_attn_patch`)
  intentionally left unset → defaults to disabled, same race-condition
  workaround as the other MI308/MI350 examples.

## Known issues

1. **MI308 (gfx942) not officially validated upstream.** vLLM/AMD only validate
   MI300X/325X/355X. MI308 shares the gfx942 ISA so the same code paths
   should run, but no upstream report confirms gpt-oss-120b on MI308.
   First run is the validation.
2. **Eagle3 aux layer indices `[2, 18, 33]` correspond to gpt-oss layer types
   `sliding/full/full`** per the model's alternating `layer_types` config.
   If draft acceptance is unexpectedly low, sampling only from full-attention
   layers might help — would require a code change in
   `lumenrl/engine/inference/vllm_teacher_engine.py:104`.

## Reference

- [openai/gpt-oss-120b on Hugging Face](https://huggingface.co/openai/gpt-oss-120b)
- [vLLM gpt-oss recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
- Sibling examples: `examples/Qwen3_8B_SDDD_MI308_vLLM/`, `examples/Kimi_K25_SDDD_MI350_vLLM/`
