# utils/cleanup.py

import gc
import multiprocessing

import torch


def cleanup_after_stage():
    """
    Cleanup between TRAIN and EVAL when running --stage all.

    - Empties CUDA cache
    - Runs Python garbage collector
    - Terminates stray multiprocessing children
    """
    print("[CLEANUP] Starting stage cleanup...")

    # GPU memory
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Python objects
    gc.collect()

    # Kill any leftover multiprocessing children
    try:
        children = multiprocessing.active_children()
        for p in children:
            print(f"[CLEANUP] Terminating stray process PID={p.pid}")
            p.terminate()
    except Exception:
        pass

    print("[CLEANUP] Stage cleanup complete.\n")
