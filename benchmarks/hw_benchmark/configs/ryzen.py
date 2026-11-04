"""Ryzen AI config: Lemonade (llama.cpp/GGUF) over its OpenAI-compatible API.

Only llama.cpp/GGUF models are used: Lemonade's ryzenai-llm (NPU/Hybrid) recipe
does not honor response_format/json_schema, so it can't do CODA's constrained
decoding regardless of speed.

One-time Lemonade-side setup this config assumes is already done:
  lemonade alias add whisper-1 Whisper-Small
  (run_bench loads each model itself with CTX_SIZE, --reasoning off, and the
  llama.cpp backend; --reasoning off keeps models from burning context on
  chain-of-thought before emitting the JSON answer)

Excluded, with reasons:
  Qwen2.5-7B-Instruct-NPU/-Hybrid, gpt-oss-20b-NPU  - ryzenai-llm ignores json_schema (0% valid).
      gpt-oss-20b-NPU also took 242s and got stuck in a reasoning loop
  gpt-oss-20b-mxfp4-GGUF  - Lemonade catalog bug: this id resolves to a speculative-decoding
      draft model, so use gpt-oss-20b-GGUF-MXFP4 (in MODELS below) instead
  DeepSeek-Qwen3-8B-GGUF  - R1-distill reasons at length (>3min/call) even with --reasoning off
  qwen3-8b-FLM, gemma4-it-12b-FLM  - FastFlowLM (NPU): ~116s/call for Qwen3-8B, Gemma-4-12B failed

The exception is MedGemma-4B-CHAMPS-FT (backend champs_finetuned): CODA's LoRA
classifier scores a fixed cause menu by likelihood instead of generating JSON, so it
runs in-process through PyTorch rather than Lemonade. Its "id" is the adapter path
(None uses inference.champs_finetuned.adapter_path from CODA's settings).

Every Lemonade and champs_finetuned model needs a "device": a llama.cpp backend from
LLAMACPP_ALLOWED_DEVICES, or a torch device from TORCH_ALLOWED_DEVICES (rocm is the
iGPU and needs a ROCm torch build). --llm-devices runs each model on whichever of
the listed devices it supports instead.
"""

STT_BACKEND = "lemonade-whisper"
LEMONADE_BASE = "http://localhost:13305/api/v1"

CTX_SIZE = 8192
LLAMACPP_ALLOWED_DEVICES = ("cpu", "vulkan", "rocm")
TORCH_ALLOWED_DEVICES = ("cpu", "rocm")

MODELS = [
    {"name": "Qwen2.5-7B-Instruct-GGUF", "backend": "lemonade",
     "id": "Qwen2.5-7B-Instruct-GGUF-Q4_K_M", "device": "vulkan"},
    {"name": "gpt-oss-20b-GGUF", "backend": "lemonade", "id": "gpt-oss-20b-GGUF-MXFP4", "device": "vulkan"},
    {"name": "Qwen3-8B-GGUF", "backend": "lemonade", "id": "Qwen3-8B-GGUF", "device": "vulkan"},
    {"name": "Qwen3-30B-A3B-GGUF", "backend": "lemonade", "id": "Qwen3-30B-A3B-GGUF", "device": "vulkan"},
    {"name": "Gemma-4-26B-A4B-it-GGUF", "backend": "lemonade", "id": "Gemma-4-26B-A4B-it-GGUF", "device": "rocm"},
    {"name": "Qwen3-14B-GGUF", "backend": "lemonade", "id": "Qwen3-14B-GGUF", "device": "vulkan"},
    {"name": "Gemma-4-12B-it-GGUF", "backend": "lemonade", "id": "Gemma-4-12B-it-GGUF", "device": "rocm"},
    {"name": "MedGemma-4B-CHAMPS-FT", "backend": "champs_finetuned", "id": None,
     "device": "rocm"},
]
