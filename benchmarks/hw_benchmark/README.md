# CODA hardware benchmark

Benchmarks both stages of CODA's pipeline on a laptop, the way CODA runs them, so
results from different machines are directly comparable. One shared codebase runs
on Apple Silicon (Macs) and Windows AMD Ryzen AI boxes; only the per-platform model
list and serving backend differ, and those are isolated in `configs/`.

1. **Speech-to-text** — WhisperLiveKit fed the shared clip as real-time PCM, the
   same streaming path CODA uses. Reports `keep_up` = wall / clip: the audio is
   played at real time, so `~1.0` means the machine keeps up in real time and
   `> 1.0` means it falls behind (the excess is trailing transcription lag).
2. **Inference** — cause-of-death across the config's models using CODA's request:
   the CHAMPS system prompt, `COD_OUTPUT_SCHEMA`, and schema-constrained decoding.

The CHAMPS prompt and schema (`coda_snapshot.py`, which reads CODA's own
`src/coda/inference/champs_prompted/prompts` and `src/coda/resources/champs`),
narratives, and audio clip are shared and identical
across machines. `keep_up` and inference latency are backend-agnostic, so Ollama/MLX
(Mac) and Lemonade (Ryzen) land on the same axes.

## Configs (how mac and ryzen differ)

`run_bench.py` auto-selects the config from the OS (`--config mac|ryzen` overrides):

| | STT backend | Inference backend | Models |
|---|---|---|---|
| **mac** (`configs/mac.py`) | mlx-whisper | Ollama + mlx-openai-server | qwen2.5-7b, gpt-oss-20b, Qwen3-30B-A3B (MLX) |
| **ryzen** (`configs/ryzen.py`) | lemonade-whisper | Lemonade (llama.cpp/GGUF) | Qwen2.5-7B, Qwen3-8B, Qwen3-14B, Gemma-4-12B |

Ryzen uses only GGUF models: Lemonade's ryzenai-llm (NPU/Hybrid) recipe does not
honor `response_format`/`json_schema`, so it can't do CODA's constrained decoding.
See the docstrings in `configs/ryzen.py` for the excluded-model rationale.

## Setup

- **Mac**: `bash setup.sh` (needs Ollama; pulls the Ollama models, installs deps).
- **Ryzen**: `powershell -ExecutionPolicy Bypass -File setup.ps1` (needs Lemonade
  already installed and models pulled; see `configs/ryzen.py` for the exact ids and
  the one-time `lemonade alias`/`lemonade load` setup).

Both install into a venv at `~/.virtualenvs/coda-inbench`.

## Run

```bash
# mac
~/.virtualenvs/coda-inbench/bin/python run_bench.py
# ryzen
%USERPROFILE%\.virtualenvs\coda-inbench\Scripts\python.exe run_bench.py
```

Runs the STT stage (several real-time reps, averaged), then each model over the 10
narratives (warm call first, then timed). Prints per-model `valid_rate` and warm
latency stats, and writes one report to the shared `results/` folder tagged with the
machine's hardware.

### Options
- `--config mac|ryzen` — override auto-detection.
- `--models NAME ...` — run a subset of the config's models.
- `--narratives path.json` — your own narratives (a JSON list of strings).
- `--audio path.wav` — your own 16 kHz mono clip for the STT stage.
- `--stt-reps N` — real-time STT passes to average (default 3).
- `--skip-stt` — inference only.
- `--skip-llm` — speech-to-text only.
- `--test-stt-backends` — run lemonade-whisper on each whisper.cpp backend (vulkan, rocm, npu, cpu).
- `--llm-devices NAME ...` — run each model on whichever of these devices it
  supports (llama.cpp: vulkan/rocm/cpu, torch: rocm/cpu) instead of its configured
  `device`, e.g. `--llm-devices vulkan rocm cpu` for a full sweep. Each device is
  reported (`llm_device`) and plotted separately.

## Plot

```bash
python plot_results.py
```

Reads every `results/*.json` and writes `comparison.png` + `summary.md`:
- **Inference**: mean latency vs memory bandwidth, colored by model family, marker
  shape by vendor. `Qwen2.5-7B` runs on every machine, so its series is the direct
  apples-to-apples hardware comparison; other families show where a machine's own
  model set lands. Best measurement per (machine, model) is used.
- **Speech-to-text**: real-time keep-up per machine, sorted by bandwidth.

## Fine-tuned MedGemma (`champs_finetuned` backend)

`MedGemma-4B-CHAMPS-FT` in `configs/ryzen.py` is CODA's LoRA fine-tune of
`google/medgemma-4b-it`. It runs through `coda.inference.champs_finetuned` in-process
(PyTorch) rather than Lemonade: it scores a fixed 60-cause menu by likelihood instead
of generating JSON. Per-case latency is the same "narrative in, ranked causes out"
measure as the other models; the one-time model load + adapter merge is reported
separately as `load_sec`. It runs on `rocm` (the iGPU) or `cpu`, set by the config
entry's `device` or `--llm-devices`.

Setup (once): install the Microsoft C++ Build Tools (gilda's `adeft` dependency
builds from source), run `setup.ps1` with `$env:CODA_INBENCH_FINETUNED = "1"` to
install CODA with its `champs-finetuned` extra and AMD's ROCm torch for gfx1151,
accept the `google/medgemma-4b-it` license on Hugging Face and `huggingface-cli login`,
and set `inference.champs_finetuned.adapter_path` in `config/settings.yaml` (or put
the path in the config entry's `id`). Then:

```
%USERPROFILE%\.virtualenvs\coda-inbench\Scripts\python.exe run_bench.py --skip-stt --models MedGemma-4B-CHAMPS-FT --llm-devices rocm
```

Caveats: it runs at bf16 in PyTorch, not Q4 in llama.cpp, and does no generation, so
its latency reflects the classifier approach as CODA deploys it, not the model on the
same engine. PyTorch has no Vulkan backend here, so compare it against each Lemonade
model's fastest valid llama.cpp backend. The benchmark narratives have no decedent
profile, so the prompt's evidence/clinical sections are empty.

## Comparability caveats
- **Quantization** differs (Mac MLX-4bit / Ollama-Q4 vs Ryzen GGUF-Q4); the exact
  model `id` is in each report. `valid_rate` guards whether constrained decoding held.
- **STT weights** differ across backends (mlx-whisper / whisper.cpp / faster-whisper),
  so `keep_up` compares throughput, not word error rate.
- **Bandwidth axis**: `hwinfo.py` looks up peak memory bandwidth per chip (neither
  Apple nor AMD expose it at runtime), so every point plots on one shared axis.

## Files
- `run_bench.py` — the benchmark driver (config-driven; STT + inference)
- `configs/` — per-platform model list + backends (`mac.py`, `ryzen.py`)
- `coda_snapshot.py` — CHAMPS request builder + schema; reads CODA's own CHAMPS prompts and allowed causes
- `stt_bench.py` — speech-to-text stage (all backends; also runnable standalone)
- `hwinfo.py` — cross-platform hardware probe, tags every report
- `plot_results.py` — charts all results together
- `narratives.json`, `assets/clip.wav` — shared inputs
- `results/` — one JSON per run, from every machine
- `setup.sh` / `setup.ps1` — mac / ryzen dependency installers
