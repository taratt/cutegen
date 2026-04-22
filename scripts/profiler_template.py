# === profiler_template.py ===
# This script is executed INSIDE Nsight Compute.
# All placeholders are replaced before writing into a temporary profiler.py.

import os
import gc
import torch
import random
import torch.cuda.nvtx as nvtx
import os, sys, shutil
import re
import glob
import importlib.util
print("PYTHON =", sys.executable)
print("CWD =", os.getcwd())
print("TORCH_EXTENSIONS_DIR =", os.environ.get("TORCH_EXTENSIONS_DIR"))
print("CUDA_HOME =", os.environ.get("CUDA_HOME"))
print("PATH =", os.environ.get("PATH"))
print("which ninja =", shutil.which("ninja"))
print("which nvcc =", shutil.which("nvcc"))
# --- placeholders that will be replaced ---
BUILD_DIR = "__BUILD_DIR__"
DEVICE_INDEX = __DEVICE_INDEX__
SEED = __SEED__
NUM_WARMUPS = __NUM_WARMUPS__
NUM_ITERS = __NUM_ITERS__
REF_SRC = """__REF_SRC__"""
GEN_SRC = """__GEN_SRC__"""
PREBUILT_SO_FILES = __PREBUILT_SO_FILES__
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

def _patch_gen_src_to_use_prebuilt_extensions(gen_src: str):
    """
    Replace every:
        some_var = load_inline(...)
    with:
        some_var = PREBUILT_EXTS["extension_name"]

    Returns:
        patched_src, ext_names
    """
    pattern = re.compile(
        r'^(?P<indent>[ \t]*)(?P<var>[A-Za-z_]\w*)\s*=\s*load_inline\((?P<body>.*?)^\)',
        re.MULTILINE | re.DOTALL,
    )

    ext_names = []

    def repl(match):
        indent = match.group("indent")
        var_name = match.group("var")
        body = match.group("body")

        name_match = re.search(r'name\s*=\s*[\'"]([^\'"]+)[\'"]', body)
        if not name_match:
            raise RuntimeError(f"Could not find extension name=... inside load_inline for variable {var_name}")

        ext_name = name_match.group(1)
        ext_names.append(ext_name)
        return f'{indent}{var_name} = PREBUILT_EXTS["{ext_name}"]'

    patched_src = pattern.sub(repl, gen_src)
    return patched_src, ext_names


def _load_prebuilt_extensions(build_dir: str, ext_names):
    """
    Load already-built .so files from build_dir.
    """
    prebuilt = {}

    for ext_name in ext_names:
        matches = glob.glob(os.path.join(build_dir, "**", f"{ext_name}.so"), recursive=True)
        if not matches:
            raise RuntimeError(f"Prebuilt extension not found for {ext_name} under {build_dir}")

        so_path = matches[0]
        spec = importlib.util.spec_from_file_location(ext_name, so_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for {so_path}")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        prebuilt[ext_name] = mod

    return prebuilt

def _patch_gen_src_to_use_prebuilt_modules(gen_src: str):
    """
    Replace:
        some_var = load_inline(...)
    with:
        some_var = PREBUILT_EXTS.pop(0)
    """
    pattern = re.compile(
        r'^(?P<indent>[ \t]*)(?P<var>[A-Za-z_]\w*)\s*=\s*load_inline\((?P<body>.*?)^\)',
        re.MULTILINE | re.DOTALL,
    )

    def repl(match):
        indent = match.group("indent")
        var_name = match.group("var")
        return f"{indent}{var_name} = PREBUILT_EXTS.pop(0)"

    return pattern.sub(repl, gen_src)

def _load_prebuilt_extension(so_files):
    mods = []
    for so_path in so_files:
        # Use the real module name from the filename
        module_name = os.path.basename(so_path).split(".")[0]

        spec = importlib.util.spec_from_file_location(
            module_name,
            so_path,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for {so_path}")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mods.append(mod)

    return mods
def load_custom_model(gen_src: str, context: dict, build_dir: str):
    os.environ["TORCH_EXTENSIONS_DIR"] = build_dir

    if not PREBUILT_SO_FILES:
        raise RuntimeError(f"No prebuilt .so files were passed into profiler.py")

    patched_src = _patch_gen_src_to_use_prebuilt_modules(gen_src)
    prebuilt_exts = _load_prebuilt_extension(PREBUILT_SO_FILES)

    wrapped_src = (
        "import os\n"
        "import gc\n"
        "import torch\n"
        f"os.environ['TORCH_EXTENSIONS_DIR'] = r'{build_dir}'\n"
        + patched_src +
        "\ntorch.cuda.synchronize()\n"
        "torch.cuda.empty_cache()\n"
        "gc.collect()\n"
    )

    context["PREBUILT_EXTS"] = prebuilt_exts

    compile(wrapped_src, "<string>", "exec")
    exec(wrapped_src, context)

    ModelNew = context.get("ModelNew")
    if ModelNew is None:
        raise RuntimeError("ModelNew missing in generated code")
    return ModelNew
#def load_custom_model(gen_src: str, context: dict, build_dir: str):
 #   os.environ["TORCH_EXTENSIONS_DIR"] = build_dir

  #  wrapped_src = (
   #     "import os\nimport gc\n"
    #    f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
     #   f"BUILD_DIRECTORY = r'{build_dir}'\n"
      #  + gen_src +
       # "\ntorch.cuda.synchronize()\n"
#        "torch.cuda.empty_cache()\n"
 #       "gc.collect()\n"
  #  )

   # compile(wrapped_src, "<string>", "exec")
#    exec(wrapped_src, context)

 #   ModelNew = context.get("ModelNew")
  #  if ModelNew is None:
   #     raise RuntimeError("ModelNew missing in generated code")
   # return ModelNew

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
