"""Benchmark CODA's speech-to-text and cause-of-death inference on a laptop.

Runs the two stages of CODA's pipeline the way CODA runs them, identically on every
machine so results are comparable:
  1. Speech-to-text: WhisperLiveKit fed the shared clip as real-time PCM.
  2. Inference: the CHAMPS system prompt, COD_OUTPUT_SCHEMA, and schema-constrained
     decoding, across the config's models. The champs_finetuned backend instead runs
     CODA's fine-tuned MedGemma classifier in-process (menu prompt, candidate
     scoring, no generation), exactly as coda.inference.champs_finetuned does.

The CHAMPS prompt and schema (coda_snapshot.py, reading CODA's own resources) plus
narratives and clip are shared. Only the per-platform model list and serving backend differ,
and those live in configs/. The runner auto-selects mac vs ryzen from the OS
(override with --config). Writes one report per run to the shared results/ folder,
tagged with the machine's hardware, for plot_results.py to chart together.
"""
import argparse
import asyncio
import json
import re
import statistics as stats
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import coda_snapshot
import stt_bench
from configs import load as load_config
from hwinfo import hardware_info
from stt_bench import run_stream as stt_run_stream

HERE = Path(__file__).resolve().parent
MLX_PORT = 8080


def wait_ready(url, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def start_mlx_server(model_id):
    subprocess.run(["pkill", "-f", "mlx-openai-server"], capture_output=True)
    time.sleep(2)
    proc = subprocess.Popen(
        ["mlx-openai-server", "launch", "--model-path", model_id,
         "--model-type", "lm", "--port", str(MLX_PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_ready(f"http://localhost:{MLX_PORT}/v1/models", 180):
        proc.terminate()
        raise RuntimeError("mlx-openai-server did not become ready")
    return proc

def lemonade_load_model(model_id, backend=None, ctx_size=None, engine="llamacpp"):
    """Load a model in Lemonade, optionally pinning the engine's backend.

    engine is "llamacpp" for LLMs or "whispercpp" for Whisper; backend is e.g.
    vulkan/rocm/cpu (or npu for whispercpp). None leaves Lemonade's default.
    """
    subprocess.run(["lemonade", "unload"], capture_output=True)
    cmd = ["lemonade", "load", model_id]
    if engine == "llamacpp":
        cmd += ["--llamacpp-args", "--reasoning off"]
        if ctx_size:
            cmd += ["--ctx-size", str(ctx_size)]
    if backend:
        cmd += [f"--{engine}", backend]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr).strip()[-200:])

def load_champs_finetuned(adapter_path, torch_device):
    """Load CODA's MedGemma CHAMPS agent on rocm (the iGPU) or cpu.

    Imported lazily so the other backends never need torch or CODA installed.
    ROCm builds of torch expose the GPU through the cuda device type.
    """
    from coda.inference.champs_finetuned.agent import (
        create_champs_finetuned_agent, finetuned_config)
    from coda.inference.champs_finetuned.engine import MedGemmaModel
    adapter_path = adapter_path or finetuned_config().adapter_path
    device_map = {"rocm": "cuda", "cpu": "cpu"}[torch_device]
    model = MedGemmaModel(adapter_path=adapter_path, device_map=device_map)
    return create_champs_finetuned_agent(adapter_path, model=model)


def champs_finetuned_infer(agent, narrative):
    """Score the cause menu for one narrative, shaped like the JSON backends' response."""
    scores = agent.score_menu(narrative)
    code = agent.menu_codes[scores.index(max(scores))]
    return {"top_causes": [{"cause_name": agent.menu[code]}]}


def infer_once(model, system, user, lemonade_base):
    backend = model["backend"]
    if backend == "ollama":
        return coda_snapshot.ollama_infer(model["id"], system, user,
            think=model.get("think"))
    if backend == "mlx":
        return coda_snapshot.openai_infer(model["id"], system, user,
            base_url=f"http://localhost:{MLX_PORT}/v1")
    return coda_snapshot.openai_infer(model["id"], system, user,
        base_url=lemonade_base)


def run_label(name, backend):
    return f"{name} [{backend}]" if backend else name


def top_cause(resp):
    causes = resp.get("top_causes") or []
    return causes[0]["cause_name"] if causes else None


def bench_model(model, narratives, system, lemonade_base, llm_device, ctx_size):
    result = {"name": model["name"], "backend": model["backend"],
              "llm_device": llm_device, "id": model["id"], "cases": []}
    proc = None
    if model["backend"] == "mlx":
        proc = start_mlx_server(model["id"])
    if model["backend"] == "lemonade":
        lemonade_load_model(model["id"], llm_device, ctx_size)
    agent = None
    if model["backend"] == "champs_finetuned":
        # Model load + adapter merge happens once, so it is kept out of latency
        t0 = time.time()
        agent = load_champs_finetuned(model["id"], llm_device)
        result["load_sec"] = round(time.time() - t0, 2)
        result["id"] = agent.adapter_path

    def infer(narrative):
        if agent is not None:
            return champs_finetuned_infer(agent, narrative)
        return infer_once(model, system, coda_snapshot.user_prompt(narrative),
            lemonade_base)

    try:
        try:
            infer(narratives[0])
        except Exception:
            pass
        for i, narrative in enumerate(narratives):
            t0 = time.time()
            try:
                resp = infer(narrative)
                valid, top = True, top_cause(resp)
            except Exception:
                valid, top = False, None
            result["cases"].append({"i": i, "latency_sec": round(time.time() - t0, 2),
                                    "valid": valid, "top_cause": top})
    finally:
        if proc is not None:
            subprocess.run(["pkill", "-f", "mlx-openai-server"], capture_output=True)

    lat = [c["latency_sec"] for c in result["cases"] if c["valid"]]
    n = len(result["cases"])
    result["valid_rate"] = sum(c["valid"] for c in result["cases"]) / n if n else 0
    result["mean_latency_sec"] = round(stats.mean(lat), 2) if lat else None
    result["median_latency_sec"] = round(stats.median(lat), 2) if lat else None
    result["stdev_latency_sec"] = round(stats.stdev(lat), 2) if len(lat) > 1 else None
    result["min_latency_sec"] = round(min(lat), 2) if lat else None
    result["max_latency_sec"] = round(max(lat), 2) if lat else None
    return result


def run_stt(audio, reps, backend):
    runs, err = [], None
    for _ in range(reps):
        try:
            runs.append(asyncio.run(stt_run_stream(audio, "small", backend,
                "localagreement", "en")))
        except Exception as e:
            err = str(e)[:200]
            break
    if not runs:
        return {"error": err or "no runs"}
    keeps = [r["keep_up"] for r in runs if r.get("keep_up") is not None]
    return {"backend": backend, "reps": len(runs), "clip_sec": runs[0].get("clip_sec"),
            "keep_up_mean": round(stats.mean(keeps), 3) if keeps else None,
            "keep_up_median": round(stats.median(keeps), 3) if keeps else None,
            "peak_gb": round(max(r.get("peak_gb") or 0 for r in runs), 2),
            "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, choices=("mac", "ryzen"),
        help="platform config (default: auto-detect from OS)")
    ap.add_argument("--narratives", default=str(HERE / "narratives.json"))
    ap.add_argument("--audio", default=str(HERE / "assets" / "clip.wav"))
    ap.add_argument("--models", nargs="*", default=None,
        help="subset of model names to run")
    ap.add_argument("--skip-stt", action="store_true",
        help="skip the speech-to-text stage")
    ap.add_argument("--skip-llm", action="store_true",
        help="skip the LLM inference stage")
    ap.add_argument("--stt-reps", type=int, default=3,
        help="how many real-time STT passes to average")
    ap.add_argument("--llm-devices", nargs="*", default=None,
        help="run each model on whichever of these devices it supports, e.g. "
             "vulkan rocm cpu (default: each model's configured device)")
    ap.add_argument("--test-stt-backends", action="store_true",
        help="check various backends for the Whisper model in the STT step (CPU, Vulkan, ROCm, NPU)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    known = {d for allowed in cfg["allowed_devices"].values() for d in allowed}
    if args.llm_devices and not known:
        ap.error(f"--llm-devices: the {cfg['name']} config's models pick their own "
                 "device, so there is nothing to choose")
    if args.llm_devices and (unknown := set(args.llm_devices) - known):
        ap.error(f"--llm-devices: unknown {sorted(unknown)}, expected {sorted(known)}")
    lemonade_base = cfg["lemonade_base"]
    if lemonade_base:
        stt_bench.LEMONADE_BASE = lemonade_base
        if not wait_ready(f"{lemonade_base}/models", 5):
            print(f"WARNING: Lemonade not reachable at {lemonade_base}. "
                  "Start it (`lemonade serve`) and confirm the port.\n")

    narratives = json.loads(Path(args.narratives).read_text())
    system = coda_snapshot.build_system_prompt()
    hw = hardware_info()
    print(f"Config: {cfg['name']}   Hardware: {hw['chip']} {hw['ram_gb']}GB "
          f"{hw['bandwidth_gbps']}GB/s")
    print(f"Narratives: {len(narratives)}\n")

    report = {"config": cfg["name"], "hardware": hw,
              "timestamp": datetime.now().isoformat(timespec="seconds"),
              "n_narratives": len(narratives), "stt": None, "models": []}

    if not args.skip_stt:
        lemonade_stt = cfg["stt_backend"] == "lemonade-whisper"
        stt_name = f"WhisperLiveKit {cfg['stt_backend']} small"
        stt_backends = [None]
        if args.test_stt_backends:
            if lemonade_stt:
                stt_backends = ["vulkan", "rocm", "npu", "cpu"]
            else:
                print("WARNING: --test-stt-backends only applies to lemonade-whisper\n")
        for stt_backend in stt_backends:
            print(f"[speech-to-text] {run_label(stt_name, stt_backend)} x{args.stt_reps}...")
            try:
                if lemonade_stt:
                    lemonade_load_model("whisper-1", stt_backend, engine="whispercpp")
                stt = run_stt(args.audio, args.stt_reps, cfg["stt_backend"])
            except Exception as e:
                stt = {"error": str(e)[:200]}
            stt["whispercpp_backend"] = stt_backend
            if "error" in stt:
                print(f"  ERROR: {stt['error']}\n")
            else:
                print(f"  keep_up mean={stt['keep_up_mean']} median={stt['keep_up_median']} "
                    f"(~1.0 keeps up, >1.0 falls behind)  peak={stt['peak_gb']}GB\n")
            # Default run stays in report["stt"] (what plot_results reads per chip);
            # a backend sweep goes in report["stt_backends"], one entry per backend.
            if stt_backend is None:
                report["stt"] = stt
            else:
                report.setdefault("stt_backends", []).append(stt)

    models = [] if args.skip_llm else [
        m for m in cfg["models"] if not args.models or m["name"] in args.models]
    ctx_size = cfg["ctx_size"]
    for m in models:
        allowed = cfg["allowed_devices"].get(m["backend"])
        if allowed is None:
            devices = [None]  # ollama/mlx pick their own device
        elif args.llm_devices:
            devices = [d for d in args.llm_devices if d in allowed]
            if not devices:
                print(f"[{m['name']}] skipped: supports none of --llm-devices "
                      f"(only {', '.join(allowed)})\n")
        else:
            devices = [m["device"]]
        for device in devices:
            label = run_label(m["name"], device)
            print(f"[{label}] ({m['backend']})...")
            try:
                r = bench_model(m, narratives, system, lemonade_base, device, ctx_size)
            except Exception as e:
                r = {"name": m["name"], "backend": m["backend"], "llm_device": device,
                     "error": str(e)[:200]}
            report["models"].append(r)
            if "error" in r:
                print(f"  ERROR: {r['error']}\n")
            else:
                if "load_sec" in r:
                    print(f"  load={r['load_sec']}s (not in latency)")
                print(f"  valid={r['valid_rate']*100:.0f}%  mean={r['mean_latency_sec']}s  "
                    f"median={r['median_latency_sec']}s\n")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9._+-]+", "_", hw["chip"] or cfg["name"])
    out = out_dir / f"{tag}_{report['timestamp'].replace(':', '')}.json"
    out.write_text(json.dumps(report, indent=2))

    print("=" * 68)
    stt_runs = ([report["stt"]] if report["stt"] else []) + report.get("stt_backends", [])
    for stt in stt_runs:
        label = run_label(f"WhisperLiveKit {cfg['stt_backend']} small",
            stt.get("whispercpp_backend"))
        if "error" in stt:
            print(f"speech-to-text ({label}): ERROR {stt['error']}")
        else:
            print(f"speech-to-text ({label}, {stt['reps']} reps): "
                  f"keep_up={stt.get('keep_up_mean')}  peak={stt.get('peak_gb')}GB")
    print(f"{'model':<40}{'valid':>7}{'mean_s':>9}{'med_s':>7}{'sd':>6}")
    for r in report["models"]:
        label = run_label(r["name"], r.get("llm_device"))
        if "error" in r:
            print(f"{label:<40}{'ERR':>7}")
        elif r.get("mean_latency_sec") is None:
            print(f"{label:<40}{r['valid_rate']*100:>6.0f}%{'-':>9}")
        else:
            sd = r.get("stdev_latency_sec")
            print(f"{label:<40}{r['valid_rate']*100:>6.0f}%"
                  f"{r['mean_latency_sec']:>9}{r['median_latency_sec']:>7}"
                  f"{(sd if sd is not None else 0):>6}")
    print(f"\nReport written to {out}")
    print("Send that JSON file back to the group.")


if __name__ == "__main__":
    main()
