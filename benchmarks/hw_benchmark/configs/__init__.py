"""Per-platform benchmark configs (model list + STT backend).

The rest of the benchmark is shared; only these differ between machines. run_bench
auto-selects mac vs ryzen from the OS, or takes --config to override.
"""
import importlib
import platform


def load(name=None):
    if name is None:
        name = "mac" if platform.system() == "Darwin" else "ryzen"
    mod = importlib.import_module(f"configs.{name}")
    cfg = {
        "name": name,
        "stt_backend": mod.STT_BACKEND,
        "lemonade_base": getattr(mod, "LEMONADE_BASE", None),
        "ctx_size": getattr(mod, "CTX_SIZE", None),
        # Devices a model of each backend can be pinned to; ollama/mlx pick their own.
        "allowed_devices": {
            "lemonade": tuple(getattr(mod, "LLAMACPP_ALLOWED_DEVICES", ())),
            "champs_finetuned": tuple(getattr(mod, "TORCH_ALLOWED_DEVICES", ())),
        },
        "models": mod.MODELS,
    }
    for m in cfg["models"]:
        allowed = cfg["allowed_devices"].get(m["backend"])
        if allowed is not None and m.get("device") not in allowed:
            raise ValueError(f"configs/{name}.py: {m['name']} needs a \"device\" "
                             f"from {allowed}, got {m.get('device')!r}")
    return cfg
