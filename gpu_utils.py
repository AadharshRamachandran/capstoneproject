import os


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def mode() -> str:
    return os.environ.get("TNM_GPU", "auto").strip().lower()


def use_gpu_torch() -> bool:
    return cuda_available() and mode() not in ("0", "off", "cpu", "no")


def use_gpu_trees() -> bool:
    return cuda_available() and mode() in ("all", "trees", "1", "on", "yes")


def xgb_kwargs() -> dict:
    return {"device": "cuda"} if use_gpu_trees() else {}


def catboost_kwargs() -> dict:
    return {"task_type": "GPU", "devices": "0"} if use_gpu_trees() else {}


def torch_device() -> str:
    return "cuda" if use_gpu_torch() else "cpu"


def describe() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"torch {torch.__version__} | CUDA {torch.version.cuda} | {p.name} {p.total_memory / 1e9:.1f} GB | TNM_GPU={mode()}"
        return f"torch {torch.__version__} | CUDA NOT available (CPU build?) | TNM_GPU={mode()}"
    except Exception as e:
        return f"torch not importable: {e!r}"
