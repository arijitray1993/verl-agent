# Molmo2-4B Integration for verl-agent

This document describes the changes made to integrate [Molmo2-4B](https://huggingface.co/allenai/Molmo2-4B) (`trust_remote_code=True`) into the verl-agent framework for vision-language RL training.

## Overview

Molmo2 differs from the existing Qwen2-VL/Qwen3-VL pipeline in several key ways:

- **Vision embedding**: Molmo2 handles image token insertion and vision embedding internally during its forward pass — no manual `masked_scatter` or `_get_input_embeds` is needed.
- **Position IDs**: Molmo2 uses standard 1D position IDs (no 3D RoPE like Qwen-VL).
- **Model class**: Loaded via `AutoModelForImageTextToText` (not `AutoModelForVision2Seq`).
- **Remote code**: Molmo2 requires `trust_remote_code=True`; model classes are resolved from the HuggingFace Hub at runtime.

## Files Changed

### 1. Model Loading — `verl/workers/fsdp_workers.py`

- Added `AutoModelForImageTextToText` to the transformers imports.
- Added a `molmo2` branch so the correct auto-class is used when loading the model.
- Patched the Molmo2 processor with `image_token`, `image_token_id`, and `_get_num_multimodal_tokens` properties for vLLM/SGLang compatibility.
- Registered the inner model with `AutoModel` so vLLM's `from_config` path works.

### 2. Model Adapter — `verl/models/transformers/molmo2.py` (new file)

A model adapter following the same pattern as `qwen3_vl.py`, providing:

- `forward_with_normal_backend` — standard forward pass returning logits.
- `forward_with_torch_backend` — fused linear + log-prob/entropy computation.
- `forward_with_triton_backend` — triton-based linear cross-entropy.
- `get_rope_index` — returns standard 1D position IDs in the 4-dim format expected by the framework.

### 3. Monkey Patch Registration — `verl/models/transformers/monkey_patch.py`

- Registered `molmo2` in `patch_forward_with_backends()` for torch/triton backend dispatch.
- Registered `molmo2` in `apply_monkey_patch()` to replace the model's forward method with `forward_with_normal_backend`.

### 4. Image Preprocessing — `agent_system/multi_turn_rollout/rollout_loop.py`

- Added a Molmo2-specific branch in `TrajectoryCollector` for multimodal processing.
- The Molmo2 processor handles image token insertion internally, so no manual `<|vision_start|>...<|vision_end|>` substitution is needed.
- Handles left-padding/truncation to `max_prompt_length` and computes standard 1D position IDs.

### 5. Bug Fix — `agent_system/environments/env_manager.py`

- Fixed `AlfredThorEnv` to use `config_thor.yaml` instead of `config_tw.yaml` (which is for `AlfredTWEnv`).

### 6. AI2-THOR Compatibility — `agent_system/environments/env_package/alfworld/alfworld/agents/environment/alfred_thor_env.py`

- Monkey-patches `Controller.__init__` to default to `CloudRendering` when no `DISPLAY` is available (headless servers).
- Monkey-patches `ThorEnv.step` to handle string actions from newer ai2thor versions.
- Monkey-patches `ThorEnv.reset` to initialize attributes before the parent `Controller.__init__` calls `reset()`.
- Monkey-patches `Controller.start` to skip redundant server restarts.

### 7. Import Fixes

- **`verl/workers/actor/dp_actor.py`** — Wrapped `flash_attn.bert_padding` import in try/except for environments without flash-attn.
- **`verl/workers/rollout/sglang_rollout/sglang_rollout.py`** — Added fallback imports for `Tool`, `get_ip`, `get_open_port` to handle different SGLang versions.

## Training Script

A ready-to-use training script is provided at:

```bash
bash examples/gigpo_trainer/run_alfworld_molmo2.sh
```

Key configuration differences from the Qwen-based scripts:

| Setting | Value |
| --- | --- |
| `model.path` | `allenai/Molmo2-4B` |
| `model.trust_remote_code` | `True` |
| `model.use_remove_padding` | `False` |
| `data` mode | `visual` |
| `env.env_name` | `alfworld/AlfredThorEnv` |
| FSDP wrap layer | `Molmo2DecoderLayer` |
| `VLLM_ATTENTION_BACKEND` | `TORCH_SDPA` |

## Installation Notes

On top of the standard verl-agent installation, Molmo2 requires:

```bash
# Molmo2 uses trust_remote_code, so no extra model package is needed.
# Make sure transformers is recent enough to support AutoModelForImageTextToText.
pip install transformers>=4.45.0

# For ALFWorld visual (AlfredThorEnv), install ai2thor:
pip install ai2thor
```

If running on a headless server (no X display), the `CloudRendering` monkey-patch in `alfred_thor_env.py` will activate automatically. No manual configuration is needed.
