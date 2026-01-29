# === profiler_template.py ===
# This script is executed INSIDE Nsight Compute.
# All placeholders are replaced before writing into a temporary profiler.py.

import os
import gc
import torch
import random
import torch.cuda.nvtx as nvtx

# --- placeholders that will be replaced ---
BUILD_DIR = "__BUILD_DIR__"
DEVICE_INDEX = __DEVICE_INDEX__
SEED = __SEED__
NUM_WARMUPS = __NUM_WARMUPS__
NUM_ITERS = __NUM_ITERS__
REF_SRC = """__REF_SRC__"""
GEN_SRC = """__GEN_SRC__"""

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_original_model_and_inputs(ref: str, context: dict):
    compile(ref, "<string>", "exec")
    exec(ref, context)
    Model = context.get("Model")
    get_init_inputs = context.get("get_init_inputs")
    get_inputs = context.get("get_inputs")
    if Model is None or get_init_inputs is None or get_inputs is None:
        raise RuntimeError("Model / get_init_inputs / get_inputs missing")
    return Model, get_init_inputs, get_inputs

def load_custom_model(gen_src: str, context: dict, build_dir: str):
    os.environ["TORCH_EXTENSIONS_DIR"] = build_dir

    wrapped_src = (
        "import os\nimport gc\n"
        f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        + gen_src +
        "\ntorch.cuda.synchronize()\n"
        "torch.cuda.empty_cache()\n"
        "gc.collect()\n"
    )

    compile(wrapped_src, "<string>", "exec")
    exec(wrapped_src, context)

    ModelNew = context.get("ModelNew")
    if ModelNew is None:
        raise RuntimeError("ModelNew missing in generated code")
    return ModelNew

def main():
    torch.cuda.set_device(DEVICE_INDEX)
    device = torch.device(f"cuda:{DEVICE_INDEX}")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(REF_SRC, context)
    ModelNew = load_custom_model(GEN_SRC, context, BUILD_DIR)

    set_seed(SEED)
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
        for x in init_inputs
    ]

    set_seed(SEED)
    inputs = get_inputs()
    inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    with torch.no_grad():
        set_seed(SEED)
        model_new = ModelNew(*init_inputs).cuda(device=device)

        for _ in range(NUM_WARMUPS):
            model_new(*inputs)
            torch.cuda.synchronize(device=device)

        nvtx.range_push("CUTGEN_PROFILE_ITER")

        for _ in range(NUM_ITERS):
            model_new(*inputs)
            torch.cuda.synchronize(device=device)
        nvtx.range_pop()

    torch.cuda.synchronize(device=device)
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
