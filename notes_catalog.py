#!/usr/bin/env python3
"""
notes_catalog.py — Contextual notes for model JSON configuration files.

Responsibilities (one file, one purpose):
  - Define a catalog of operator-facing notes keyed on runtime context.
  - Inject only the notes that are relevant to the actual hardware, model
    family, and software stack detected at first boot.
  - Keep model JSON files clean: no RTX 5060 block-size arithmetic in a
    Gemma config running on an H100, no Qwen reasoning-parser note in a
    Mistral config, and so on.

Design principles:
  - ADDITIVE ONLY: notes are appended to whatever the JSON already holds.
    Existing manually-written notes are never modified or removed.
    Re-running on an already-configured model is therefore safe.
  - IDEMPOTENT: each note carries a short unique tag. If a note with that
    tag already exists in the JSON (from a previous run), it is skipped.
  - ORDERED: notes are written in the order they appear in CATALOG, which
    is grouped by topic for readability in the JSON.
  - NO SIDE EFFECTS: this module never writes to disk itself. It returns
    a list of strings. The caller (vllm_launcher.py) decides when and how
    to persist them.

Usage from vllm_launcher.py:
    from notes_catalog import get_notes_for_context, NoteContext

    ctx = NoteContext(
        model_id        = "Qwen3.5-2B-AWQ-4bit",
        model_family    = "qwen",          # auto-detected
        quantization    = "4bit",          # from infer_quantization()
        has_vision      = True,            # from capabilities[]
        has_reasoning   = True,            # from capabilities[] / reasoning_parser
        vllm_version    = "0.21.1rc1...",  # from environment block
        compute_cap     = "sm_120",        # from torch.cuda.get_device_capability()
        gpu_vram_mb     = 8151,            # from nvidia-smi
        gpu_name        = "NVIDIA GeForce RTX 5060 Laptop GPU",
        existing_notes  = [...],           # current notes[] from JSON
    )
    new_notes = get_notes_for_context(ctx)
    # new_notes contains only notes not already present, relevant to this context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Context dataclass — everything the catalog needs to make decisions
# ---------------------------------------------------------------------------

@dataclass
class NoteContext:
    model_id:       str
    model_family:   str            # "qwen", "gemma", "llama", "mistral", "phi", etc.
    quantization:   str            # "4bit", "8bit", "fp16", "unknown"
    has_vision:     bool
    has_reasoning:  bool           # True if reasoning_parser is set
    has_tool_calls: bool
    vllm_version:   str            # full string, e.g. "0.21.1rc1.dev111+g..."
    compute_cap:    str            # "sm_120", "sm_90", "sm_80", etc.
    gpu_vram_mb:    int            # total VRAM in MB, 0 if unknown
    gpu_name:       str            # human-readable GPU name
    existing_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Note entry — a single note with its relevance predicate
# ---------------------------------------------------------------------------

@dataclass
class _Note:
    tag:       str          # short unique identifier embedded in the note text
                            # used for idempotency check — must appear verbatim
    text:      str          # the full note text stored in JSON
    relevant:  object       # callable(NoteContext) → bool


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

def _vllm_major_minor(version_str: str):
    """Parse (major, minor) from a vLLM version string. Returns (0,0) on failure."""
    m = re.match(r"(\d+)\.(\d+)", version_str or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    return (0, 0)


def _sm_number(compute_cap: str) -> int:
    """Convert 'sm_120' → 120, 'sm_90' → 90, etc. Returns 0 on failure."""
    m = re.match(r"sm_(\d+)", compute_cap or "")
    return int(m.group(1)) if m else 0


def _vram_class(ctx: NoteContext) -> str:
    """Classify GPU VRAM tier: 'small' <16 GB, 'mid' 16–79 GB, 'large' ≥80 GB."""
    mb = ctx.gpu_vram_mb
    if mb <= 0:
        return "unknown"
    if mb < 16 * 1024:
        return "small"
    if mb < 80 * 1024:
        return "mid"
    return "large"


def _is_ampere_or_newer(ctx: NoteContext) -> bool:
    return _sm_number(ctx.compute_cap) >= 80   # A100 = sm_80


def _is_hopper_or_newer(ctx: NoteContext) -> bool:
    return _sm_number(ctx.compute_cap) >= 90   # H100 = sm_90


def _is_blackwell_or_newer(ctx: NoteContext) -> bool:
    return _sm_number(ctx.compute_cap) >= 100  # RTX 5xxx / B100 = sm_100+


def _vllm_021_or_newer(ctx: NoteContext) -> bool:
    return _vllm_major_minor(ctx.vllm_version) >= (0, 21)


def _is_mamba_family(ctx: NoteContext) -> bool:
    """Mamba/SSM architectures use non-standard block sizes in vLLM."""
    return ctx.model_family in ("qwen",)   # Qwen3.x uses mamba-hybrid layers


# ---------------------------------------------------------------------------
# CATALOG
# Each entry: tag (unique), text (stored verbatim in JSON), relevant predicate.
# Groups are for human readability only; all entries are evaluated for every boot.
# ---------------------------------------------------------------------------

CATALOG: List[_Note] = [

    # ── First-boot / template ────────────────────────────────────────────────

    _Note(
        tag="first-boot-template",
        text=(
            "Auto-generated template on first boot. "
            "Update params_b, description, license, and capabilities to match "
            "the actual model."
        ),
        relevant=lambda ctx: True,
    ),

    # ── GPU memory and utilization ───────────────────────────────────────────

    _Note(
        tag="gpu-util-safe-default",
        text=(
            "gpu_memory_utilization starts at 0.90 (safe conservative default). "
            "Raise toward 0.95–0.98 only after confirming the model loads without "
            "OOM on this hardware. Never set to 1.0 — vLLM requires headroom for "
            "CUDA graph capture."
        ),
        relevant=lambda ctx: True,
    ),

    _Note(
        tag="gpu-util-small-vram",
        text=(
            f"Small VRAM GPU detected (<16 GB). "
            "Keep gpu_memory_utilization ≤ 0.98 and monitor KV cache size after "
            "calibration. If max_model_len calibrates below 32768, consider a "
            "smaller quantisation (e.g. 4-bit instead of 8-bit) or a smaller model."
        ),
        relevant=lambda ctx: _vram_class(ctx) == "small",
    ),

    _Note(
        tag="gpu-util-large-vram",
        text=(
            "Large VRAM GPU detected (≥80 GB, e.g. H100/H200). "
            "gpu_memory_utilization can safely be set to 0.90–0.95. "
            "Leaving 5–10% headroom avoids OOM during CUDA graph capture and "
            "allows the OS and other processes to share the device. "
            "For multi-GPU (tensor parallel) deployments, set tensor_parallel_size "
            "in extra_vllm_args and divide gpu_memory_utilization by the number of "
            "GPUs only if using pipeline parallelism."
        ),
        relevant=lambda ctx: _vram_class(ctx) == "large",
    ),

    # ── CUDA allocator ───────────────────────────────────────────────────────

    _Note(
        tag="pytorch-cuda-alloc-conf",
        text=(
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set in the vLLM "
            "subprocess environment by vllm_launcher.py. This prevents CUDA "
            "allocator fragmentation OOM when the unquantised vision encoder "
            "processes large image batches in successive calls. The model weights "
            "may be quantised (e.g. AWQ 4-bit) but vision encoder projection "
            "layers typically run in fp16 and are not quantised. Without "
            "expandable_segments, reserved-but-fragmented VRAM cannot be reused, "
            "causing torch.OutOfMemoryError mid-stream. Safe on all hardware — "
            "no overhead when fragmentation is absent."
        ),
        relevant=lambda ctx: True,
    ),

    # ── max_model_len calibration ────────────────────────────────────────────

    _Note(
        tag="max-model-len-calibration",
        text=(
            "max_model_len is auto-calibrated by vllm_launcher.py after the first "
            "successful start. The launcher reads the actual KV cache token count "
            "from vllm_api.log and writes the optimal value back to this JSON. "
            "The formula creates a stable fixed point: once max_model_len equals "
            "the KV cache token count, the same VRAM profiling overhead reproduces "
            "the same KV cache on the next boot — no further updates occur. "
            "Floor is 65536."
        ),
        relevant=lambda ctx: True,
    ),

    # ── Block size / mamba ────────────────────────────────────────────────────

    _Note(
        tag="mamba-block-size",
        text=(
            "block_size is set by vLLM's mamba page-alignment logic (interface.py), "
            "NOT by max_model_len. For Qwen3.x mamba-hybrid models this is "
            "typically 544 tokens. All KV cache sizes are exact multiples of "
            "block_size. The 1-block loss at large max_model_len is caused by "
            "vLLM allocating a larger dummy batch during KV cache profiling, "
            "consuming slightly more VRAM and leaving 1 fewer block — expected."
        ),
        relevant=lambda ctx: _is_mamba_family(ctx),
    ),

    # ── vLLM CUDA graph profiling ─────────────────────────────────────────────

    _Note(
        tag="vllm-021-cuda-graph-profiling",
        text=(
            "CUDA graph memory profiling is ON by default since vLLM 0.21. "
            "gpu_memory_utilization=0.98 is equivalent to ~0.948 effective "
            "utilisation with profiling enabled. "
            "Set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 to disable "
            "(reverts to pre-0.21 behaviour). Do NOT do this — profiling gives "
            "accurate VRAM accounting."
        ),
        relevant=lambda ctx: _vllm_021_or_newer(ctx),
    ),

    # ── Quantization ─────────────────────────────────────────────────────────

    _Note(
        tag="awq-compressed-tensors",
        text=(
            "AWQ quantisation: set vllm_quantization to 'compressed-tensors'. "
            "vLLM auto-detects this from the model's config.json, but being "
            "explicit prevents unexpected fallback if the config changes. "
            "The display field 'quantization' (e.g. '4bit') is separate from "
            "'vllm_quantization' which is the vLLM backend identifier."
        ),
        relevant=lambda ctx: "awq" in ctx.model_id.lower(),
    ),

    _Note(
        tag="gguf-note",
        text=(
            "GGUF quantisation: vLLM supports GGUF via --quantization gguf. "
            "Set vllm_quantization to 'gguf' in this JSON. "
            "GGUF models use a single-file format; set 'path' to the .gguf file "
            "path, not a directory."
        ),
        relevant=lambda ctx: "gguf" in ctx.model_id.lower(),
    ),

    _Note(
        tag="fp16-no-quantization",
        text=(
            "fp16 / bf16 model: no quantisation backend needed. "
            "Leave vllm_quantization empty. "
            "On Hopper (H100) and newer GPUs, bf16 is preferred over fp16 for "
            "numerical stability. vLLM selects the dtype automatically."
        ),
        relevant=lambda ctx: ctx.quantization in ("fp16", "unknown")
                             and "gguf" not in ctx.model_id.lower()
                             and "awq"  not in ctx.model_id.lower(),
    ),

    # ── Vision / multimodal ───────────────────────────────────────────────────

    _Note(
        tag="vision-encoder-unquantised",
        text=(
            "Vision encoder warning: even when model weights are quantised "
            "(e.g. AWQ 4-bit), the vision encoder projection layers typically "
            "run in fp16 (unquantised). Peak VRAM usage during large image batches "
            "can therefore exceed what the quantised-weights baseline suggests. "
            "Monitor VRAM with nvidia-smi during multi-image requests."
        ),
        relevant=lambda ctx: ctx.has_vision and ctx.quantization == "4bit",
    ),

    _Note(
        tag="vision-limit-mm-per-prompt",
        text=(
            "limit_mm_per_prompt.image is auto-derived as "
            "min(30, vision_encoder_budget // 256). "
            "Set 'image' to a non-null integer to override. "
            "On small-VRAM GPUs, reducing this limit lowers peak VRAM during "
            "multi-page document analysis."
        ),
        relevant=lambda ctx: ctx.has_vision,
    ),

    _Note(
        tag="vision-triton-jit",
        text=(
            "Triton JIT kernels for vision (_bilinear_pos_embed_kernel and others) "
            "are compiled on the first vision call of each session. This causes a "
            "~1-2 s latency spike on the very first image/document request only. "
            "Compiled kernels are cached to ~/.cache/vllm/torch_compile_cache "
            "and are cache hits on all subsequent sessions."
        ),
        relevant=lambda ctx: ctx.has_vision,
    ),

    # ── Reasoning / chain-of-thought ─────────────────────────────────────────

    _Note(
        tag="reasoning-parser-thinking-budget",
        text=(
            "Chain-of-thought reasoning is enabled via reasoning_parser. "
            "Thinking tokens consume output budget: with safe_max=16384, a long "
            "reasoning trace can exhaust the budget before the visible answer is "
            "written. If responses truncate, increase max_output_tokens in this "
            "JSON or reduce thinking depth in the UI."
        ),
        relevant=lambda ctx: ctx.has_reasoning,
    ),

    _Note(
        tag="qwen-reasoning-parser",
        text=(
            "Qwen3 reasoning parser: use 'qwen3' for reasoning_parser and "
            "'qwen3_coder' for tool_call_parser. These are the correct vLLM "
            "parser identifiers for Qwen3.x models. Do not use 'qwen2' — it "
            "does not handle Qwen3 thinking tokens correctly."
        ),
        relevant=lambda ctx: ctx.model_family == "qwen" and ctx.has_reasoning,
    ),

    # ── Tool calls ────────────────────────────────────────────────────────────

    _Note(
        tag="tool-calls-enable-auto",
        text=(
            "enable_auto_tool_choice: true passes --enable-auto-tool-choice to "
            "vLLM, allowing the model to invoke tools freely without an explicit "
            "tool_choice='auto' in every request. Required for agent loop "
            "operation. Set to false for pure chat deployments where tool calls "
            "are undesired."
        ),
        relevant=lambda ctx: ctx.has_tool_calls,
    ),

    # ── GPU family / architecture ─────────────────────────────────────────────

    _Note(
        tag="ampere-bf16",
        text=(
            "Ampere GPU (A100 / sm_80) or newer detected. "
            "bf16 is natively supported and preferred over fp16 for training "
            "and fine-tuning. For inference, vLLM selects the optimal dtype "
            "automatically. Flash Attention 2 is available on Ampere+."
        ),
        relevant=lambda ctx: _is_ampere_or_newer(ctx)
                             and not _is_hopper_or_newer(ctx),
    ),

    _Note(
        tag="hopper-fp8",
        text=(
            "Hopper GPU (H100 / sm_90) or newer detected. "
            "FP8 quantisation is natively supported via Transformer Engine. "
            "For maximum throughput, consider fp8 quantised model variants and "
            "set vllm_quantization to 'fp8'. Flash Attention 3 is available on "
            "Hopper+. Tensor parallelism scales efficiently across H100 NVLink "
            "nodes: set tensor_parallel_size in extra_vllm_args."
        ),
        relevant=lambda ctx: _is_hopper_or_newer(ctx)
                             and not _is_blackwell_or_newer(ctx),
    ),

    _Note(
        tag="blackwell-sm120",
        text=(
            "Blackwell GPU (RTX 5xxx / B100+ / sm_100+) detected. "
            "vLLM cu132 and torch cu128 coexist via CUDA driver backward-compatible "
            "ABI — this is expected and not an error. "
            "FP4 (NVFP4) quantisation is natively supported on Blackwell; "
            "vllm_quantization 'compressed-tensors' with 4-bit weights is the "
            "current recommended path until native fp4 kernels ship in vLLM stable."
        ),
        relevant=lambda ctx: _is_blackwell_or_newer(ctx),
    ),

    # ── Multi-GPU / cluster ───────────────────────────────────────────────────

    _Note(
        tag="multi-gpu-tensor-parallel",
        text=(
            "Multi-GPU deployment: add tensor_parallel_size and "
            "pipeline_parallel_size to extra_vllm_args as needed. "
            "Example: {\"extra_vllm_args\": [\"--tensor-parallel-size\", \"8\"]}. "
            "gpu_memory_utilization applies per-GPU. "
            "For H100 clusters with NVLink, tensor parallelism is preferred over "
            "pipeline parallelism for latency-sensitive workloads."
        ),
        relevant=lambda ctx: _vram_class(ctx) == "large",
    ),

    # ── vLLM process management ───────────────────────────────────────────────

    _Note(
        tag="leaked-semaphore-shutdown",
        text=(
            "resource_tracker: 1 leaked semaphore at shutdown — vLLM-internal "
            "multiprocessing cleanup artefact, not actionable from application "
            "code. Expected and harmless."
        ),
        relevant=lambda ctx: True,
    ),

    _Note(
        tag="num-gpu-blocks-override",
        text=(
            "Overriding num_gpu_blocks=0 with num_gpu_blocks_override=512 in "
            "vLLM logs — internal calibration step, not an error."
        ),
        relevant=lambda ctx: True,
    ),

    _Note(
        tag="triton-jit-text",
        text=(
            "Triton JIT kernels for text inference (_zero_kv_blocks_kernel, "
            "_compute_slot_mapping_kernel, _causal_conv1d_fwd_kernel, "
            "_fused_post_conv_kernel) are compiled on the first inference of "
            "each session (~1-2 s latency spike on the very first request only). "
            "Cached to ~/.cache/vllm/torch_compile_cache. The 'consider extending "
            "warmup' suggestion in the vLLM log is informational — no action needed."
        ),
        relevant=lambda ctx: True,
    ),

]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_model_family(model_id: str) -> str:
    """
    Infer a normalised model family string from a model directory name.
    Used to select the right catalog entries without requiring a HuggingFace
    metadata fetch — all inference is purely from the local directory name.
    """
    lo = model_id.lower()
    if "qwen"    in lo: return "qwen"
    if "gemma"   in lo: return "gemma"
    if "llama"   in lo: return "llama"
    if "mistral" in lo: return "mistral"
    if "mixtral" in lo: return "mixtral"
    if "phi"     in lo: return "phi"
    if "falcon"  in lo: return "falcon"
    if "deepseek" in lo: return "deepseek"
    if "yi"      in lo: return "yi"
    if "command" in lo: return "command-r"
    if "gemma"   in lo: return "gemma"
    if "solar"   in lo: return "solar"
    if "orca"    in lo: return "orca"
    return "unknown"


def get_notes_for_context(ctx: NoteContext) -> List[str]:
    """
    Return a list of note strings relevant to this context and NOT already
    present in ctx.existing_notes (idempotency — safe to re-run on existing JSON).

    Each note is stored with a hidden machine-readable tag prefix:
        [tag:some-tag] Human-readable note text…

    The tag prefix is what the idempotency check searches for.  It is terse
    enough to be unobtrusive in the JSON but unique enough to never collide
    with legitimate note content.

    Notes are returned in CATALOG order (topic-grouped for readable JSON).
    """
    existing_blob = "\n".join(ctx.existing_notes)
    result: List[str] = []

    for entry in CATALOG:
        # Skip if this tag is already recorded in any existing note.
        tag_marker = f"[tag:{entry.tag}]"
        if tag_marker in existing_blob:
            continue
        # Skip if not relevant to this context.
        try:
            if not entry.relevant(ctx):
                continue
        except Exception:
            continue
        # Prepend the tag marker so future runs can detect it.
        result.append(f"{tag_marker} {entry.text}")

    return result
