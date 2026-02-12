# Molmo2-4B vLLM Integration for verl-agent

## Problem

vLLM 0.11.x only ships with Molmo v1 support. Molmo2 (`allenai/Molmo2-4B`) has a substantially different architecture and cannot reuse the v1 model code. This document describes all changes needed to run Molmo2 with vLLM 0.11.2 inside verl-agent's hybrid FSDP+vLLM training loop.

## Environment

- **vLLM**: 0.11.2
- **PyTorch**: 2.9.0+cu128
- **Transformers**: 4.57.1
- **Model**: `allenai/Molmo2-4B`

---

## Files Changed

### 1. `verl/workers/rollout/vllm_rollout/models/__init__.py` (new)

Empty package init file. Makes the `models/` directory a Python package so vLLM's `ModelRegistry` can import from it.

### 2. `verl/workers/rollout/vllm_rollout/models/molmo2.py` (new, ~1440 lines)

Full vLLM-compatible model implementation for Molmo2. This is the core of the integration.

**Why needed**: Molmo2's architecture differs from Molmo v1 in fundamental ways (see Architecture Differences below). None of the v1 code can be reused directly.

**Key components**:

| Component | Description |
|-----------|-------------|
| `Molmo2VisionTransformer` | ViT with bias in patch embedding, 729 positional embeddings (27x27), no CLS token |
| `Molmo2VisionBackbone` | Vision pipeline with `pooled_patches_idx`-based pooling (v1 uses `image_masks`) |
| `Molmo2ImageProjectorMLP` | SwiGLU projector (gate/up/down) instead of v1's Linear-GELU-Linear |
| `Molmo2Attention` | Fused QKV with per-head RMSNorm (Qwen3-style QK norm) |
| `Molmo2DecoderLayer` | **Pre-norm** residual blocks (`norm_after=false`) |
| `Molmo2ForConditionalGeneration` | Top-level model implementing `SupportsMultiModal`, `SupportsPP` |
| `Molmo2ProcessorWrapper` | Wraps HF processor for vLLM's multimodal API |
| `Molmo2ProcessingInfo` | Token count computation per image size |
| `Molmo2DummyInputsBuilder` | Dummy inputs for vLLM memory profiling |
| `Molmo2MultiModalProcessor` | Multimodal processing with prompt insertion logic |

**Weight mapping** (HF -> vLLM):
```python
WeightsMapper(
    orig_to_new_substr={
        "image_projector.w1.": "image_projector.gate_proj.",
        "att_proj": "qkv_proj",
        "attn_out": "o_proj",
        "ff_proj": "gate_up_proj",
        "ff_out": "down_proj",
        "attn_norm": "input_layernorm",
        "ff_norm": "post_attention_layernorm",
    },
    orig_to_new_prefix={
        "model.vision_backbone.": "vision_backbone.",
        "model.transformer.blocks.": "model.layers.",
        "model.transformer.ln_f.": "model.norm.",
    },
)
```

**Critical multimodal fixes for vLLM 0.11.2**:

1. **`_hf_processor_applies_updates()` must return `False`**: The Molmo2 HF processor does NOT insert image tokens into `input_ids` (unlike some other VLMs). Without this override, vLLM assumes image tokens are already present and fails validation with: `RuntimeError: Expected there to be 1 prompt placeholders corresponding to 1 image items, but instead found 0`.

2. **`_get_prompt_updates()` uses `PromptIndexTargets.start()`**: Image tokens are inserted at the beginning of the prompt. Using `prefix("<|im_end|>")` or `prefix("<|endoftext|>")` fails because:
   - `<|im_end|>` (token 151645) is the EOS token, not the same as `<|endoftext|>` (token 151643)
   - The tokenized prompt doesn't start with either during profiling
   - `PromptIndexTargets.start()` always matches at position 0, working for both profiling and inference

### 3. `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` (modified)

Added model registration before any vLLM engine initialization:

```python
from vllm.model_executor.models import ModelRegistry

ModelRegistry.register_model(
    "Molmo2ForConditionalGeneration",
    "verl.workers.rollout.vllm_rollout.models.molmo2:Molmo2ForConditionalGeneration"
)
```

**Placement matters**: This must execute at import time, BEFORE `LLM(...)` is instantiated. When vLLM reads the HF config and sees `model_type: "molmo2"` / `architectures: ["Molmo2ForConditionalGeneration"]`, it looks up the registry to find the implementation class.

### 4. `examples/gigpo_trainer/run_alfworld_molmo2.sh` (modified)

- Changed `ENGINE=sglang` to `ENGINE=vllm`
- Set `actor_rollout_ref.model.path=allenai/Molmo2-4B`
- Added FSDP wrap policy for `Molmo2DecoderLayer`

---

## Architecture Differences: Molmo2 vs Molmo v1

| Feature | Molmo v1 | Molmo2-4B |
|---------|----------|-----------|
| Config structure | Flat | Nested (`vit_config`, `text_config`, `adapter_config`) |
| ViT class token | 1 (prefix token) | **0** (no CLS token) |
| ViT patch embedding | No bias | **Has bias** |
| Image position embeddings | 577 (24x24 + CLS) | **729** (27x27) |
| Vision pooling | `image_masks` (binary) | **`pooled_patches_idx`** (integer indices) |
| Image projector | Linear -> GELU -> Linear | **SwiGLU** (gate/up/down) |
| Decoder norm order | Post-norm | **Pre-norm** (`norm_after=false`) |
| QK normalization | None | **Per-head RMSNorm** (`qk_norm_type="qwen3"`) |
| HF weight nesting | Flat `att_proj`, `ff_proj` | Nested under `self_attn.att_proj`, `mlp.ff_proj` |
| Special image tokens | `<|im_patch|>` format | `<im_patch>` format (no pipes, different token IDs) |
| EOS token | `<|endoftext|>` | `<|im_end|>` (ChatML format) |

---

## Debugging Notes

### Common errors and solutions

**`RuntimeError: Expected there to be N prompt placeholders ... but instead found 0`**

The multimodal profiling system can't find image token placeholders. Causes:
- `_hf_processor_applies_updates()` returns `True` (default) but the HF processor didn't insert image tokens -> Override to return `False`
- `_get_prompt_updates()` target doesn't match the tokenized prompt -> Use `PromptIndexTargets.start()`

**`AssertionError: Failed to apply prompt replacement for mm_items['image'][0]`**

The insertion target in `_get_prompt_updates()` doesn't match any position in the prompt. The `prefix()` target requires the prompt to literally start with that string after tokenization. Use `start()` instead.

**`WARNING: The legacy code for batching multi-modal kwargs is deprecated`**

Harmless warning. To suppress, add `merge_by_field_config = True` to the model class (not yet implemented).

### Token ID reference for Molmo2-4B

| Token | ID | Notes |
|-------|------|-------|
| `<|endoftext|>` | 151643 | NOT the EOS token in Molmo2 |
| `<|im_start|>` | 151644 | ChatML start |
| `<|im_end|>` | 151645 | EOS/BOS token |
| `<im_start>` | 151936 | High-res image block start |
| `<im_end>` | 151937 | Image block end |
| `<im_patch>` | 151938 | Image patch token |
| `<im_col>` | 151939 | Column separator |
| `<low_res_im_start>` | 151940 | Low-res block start |
| `<\|image\|>` | 151941 | Image placeholder |
| `<im_low>` | 151942 | Low-res token |
