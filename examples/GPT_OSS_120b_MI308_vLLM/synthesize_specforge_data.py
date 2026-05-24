"""SpecForge-style training data synthesis for GPT-OSS-120B Eagle3.

Re-generates assistant responses for prompts taken from two upstream
datasets, using the same teacher (GPT-OSS-120B) that will run at inference
time. Without this, training Eagle on the original GPT-3.5 / Llama
responses creates a train/inference distribution mismatch and tanks the
speculative acceptance rate.

Output: one JSONL file per source dataset under ``--output-dir``. Each
record has the schema consumed by
``lumenrl/trainer/spec_distill_trainer.py:_get_batch_sequences`` ::

    {"conversations": [
        {"role": "user", "content": "<original prompt>"},
        {"role": "assistant", "content": "<teacher-generated response>"}
    ]}

Resumable: re-running the command counts existing output lines and skips
that many prompts. The datasets are Arrow-backed and return rows in
stable order, so an int counter is sufficient. ``/dev/shm`` is tmpfs and
is wiped on host reboot — copy outputs to durable storage before reboot.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source dataset registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DatasetSpec:
    hf_name: str
    split: str
    prompt_extractor: Callable[[Dict[str, Any]], str]
    out_filename: str


def _extract_ultrachat_prompt(sample: Dict[str, Any]) -> str:
    return sample["prompt"]


def _extract_magpie_prompt(sample: Dict[str, Any]) -> str:
    raw = sample.get("instruction")
    if raw:
        return raw
    convs = sample.get("conversations") or []
    if convs and isinstance(convs[0], dict):
        return convs[0].get("value", "") or convs[0].get("content", "")
    return ""


SOURCES: Dict[str, _DatasetSpec] = {
    "ultrachat": _DatasetSpec(
        hf_name="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        prompt_extractor=_extract_ultrachat_prompt,
        out_filename="ultrachat_200k_synth.jsonl",
    ),
    "magpie": _DatasetSpec(
        hf_name="Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        split="train",
        prompt_extractor=_extract_magpie_prompt,
        out_filename="magpie_300k_synth.jsonl",
    ),
}


# ---------------------------------------------------------------------------
# Environment setup (must happen BEFORE importing vllm)
# ---------------------------------------------------------------------------


_TORCHRUN_PREFIXES = (
    "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK",
    "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
    "TORCHELASTIC_", "TORCH_NCCL_", "NCCL_ASYNC",
    "OMP_NUM_THREADS",
)


def _strip_torchrun_env() -> None:
    for key in list(os.environ.keys()):
        if any(key.startswith(p) for p in _TORCHRUN_PREFIXES):
            del os.environ[key]


def _configure_gpus_and_atom(gpu_ids: str, quantization: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    if not quantization:
        os.environ["ATOM_DISABLE_VLLM_PLUGIN"] = "1"
        os.environ["VLLM_PLUGINS"] = ""
        logger.info("ATOM plugin disabled (vLLM native MXFP4 autodetect)")
    else:
        logger.info("ATOM plugin active for %s quantization", quantization)


# ---------------------------------------------------------------------------
# vLLM construction (late import — CUDA env must already be set)
# ---------------------------------------------------------------------------


def _build_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    engine_kwargs: Dict[str, Any] = dict(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        compilation_config={"cudagraph_mode": "NONE"},
        seed=args.seed,
    )

    if args.quantization in ("mxfp4", "fp4"):
        engine_kwargs["quantization"] = "mxfp4"
        engine_kwargs["kv_cache_dtype"] = "fp8_e4m3"
        logger.info("MXFP4 explicit quantization enabled")
    elif args.quantization:
        engine_kwargs["quantization"] = args.quantization

    logger.info(
        "Building vLLM (tp=%d, max_model_len=%d, max_num_seqs=%d, quant=%s)",
        args.tp_size, args.max_model_len, args.max_num_seqs,
        args.quantization or "autodetect",
    )
    return LLM(**engine_kwargs)


def _build_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# Per-dataset synthesis loop
# ---------------------------------------------------------------------------


def _count_existing_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _synthesize_one(
    llm: Any,
    tokenizer: Any,
    sampling_params: Any,
    spec: _DatasetSpec,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    from datasets import load_dataset

    logger.info("[%s] loading dataset split=%s", spec.hf_name, spec.split)
    ds = load_dataset(spec.hf_name, split=spec.split)
    total = len(ds) if args.per_dataset_limit == 0 else min(
        args.per_dataset_limit, len(ds)
    )

    already = _count_existing_lines(out_path)
    if already >= total:
        logger.info("[%s] already complete (%d / %d) — skipping",
                    spec.hf_name, already, total)
        return
    if already > 0:
        logger.info("[%s] resuming at prompt %d / %d",
                    spec.hf_name, already, total)
    else:
        logger.info("[%s] starting fresh: %d prompts", spec.hf_name, total)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out_f:
        for batch_start in range(already, total, args.batch_prompts):
            batch_end = min(batch_start + args.batch_prompts, total)

            raw_prompts: List[str] = [
                spec.prompt_extractor(ds[i]) for i in range(batch_start, batch_end)
            ]
            rendered: List[str] = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in raw_prompts
            ]

            outputs = llm.generate(rendered, sampling_params, use_tqdm=False)

            for raw, out in zip(raw_prompts, outputs):
                assistant_text = out.outputs[0].text
                record = {
                    "conversations": [
                        {"role": "user", "content": raw},
                        {"role": "assistant", "content": assistant_text},
                    ],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            logger.info("[%s] %d / %d (+%d)",
                        spec.hf_name, batch_end, total, batch_end - batch_start)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SpecForge-style synthesis for GPT-OSS-120B Eagle3 training.",
    )
    parser.add_argument("--model", default="/dev/shm/gpt-oss-120b")
    parser.add_argument("--output-dir", default="/dev/shm/gpt_oss_120b_synth")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--quantization", default="",
        help='Pass "" (default) to let vLLM autodetect MXFP4 from the checkpoint.',
    )
    parser.add_argument(
        "--per-dataset-limit", type=int, default=0,
        help="0 = no cap. Use small values (e.g. 100) for smoke tests.",
    )
    parser.add_argument(
        "--datasets", default="ultrachat,magpie",
        help="Comma-separated subset of: " + ", ".join(SOURCES.keys()),
    )
    parser.add_argument(
        "--batch-prompts", type=int, default=4096,
        help="Prompts per llm.generate(...) call (keeps continuous-batch saturated).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()

    selected: List[str] = [s.strip() for s in args.datasets.split(",") if s.strip()]
    for name in selected:
        if name not in SOURCES:
            raise SystemExit(
                f"Unknown dataset {name!r}. Choices: {sorted(SOURCES)}"
            )

    _strip_torchrun_env()
    _configure_gpus_and_atom(args.gpu_ids, args.quantization)

    from transformers import AutoTokenizer

    logger.info("Loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = _build_llm(args)
    sampling_params = _build_sampling_params(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    for name in selected:
        spec = SOURCES[name]
        out_path = output_dir / spec.out_filename
        _synthesize_one(llm, tokenizer, sampling_params, spec, args, out_path)

    logger.info("Synthesis complete.")


if __name__ == "__main__":
    main()
