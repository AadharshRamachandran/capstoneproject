
import sys, json, inspect, multiprocessing


def main():
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    import torch
    print("[ts_worker] torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
          "| requested device:", a.get("device"), flush=True)
    from totalsegmentator.python_api import totalsegmentator
    sup = set(inspect.signature(totalsegmentator).parameters)
    cand = dict(fast=a["fast"], force_split=a["force_split"], nr_thr_resamp=a["threads"], nr_thr_saving=a["threads"],
                quiet=False, verbose=False, statistics=False, preview=False)
    if a.get("device"):
        cand["device"] = a["device"]
    print("[ts_worker] starting segmentation (first run may download model weights)", flush=True)
    totalsegmentator(a["inp"], a["out"], **{k: v for k, v in cand.items() if k in sup})
    print("[ts_worker] finished", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
