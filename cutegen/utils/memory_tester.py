import gc
import time
import torch
import torch.nn as nn

# -----------------------------
# Original model/task
# -----------------------------
class Model(nn.Module):
    """
    Simple model that performs a Tanh activation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

batch_size = 4096
dim = 393216
dtype = torch.float32
device = "cuda"


# -----------------------------
# Helpers
# -----------------------------
def gb(x: int) -> float:
    return x / (1024 ** 3)

def tensor_nbytes(shape, dtype):
    numel = 1
    for s in shape:
        numel *= s
    return numel * torch.tensor([], dtype=dtype).element_size()

def print_mem(prefix=""):
    free_b, total_b = torch.cuda.mem_get_info()
    alloc_b = torch.cuda.memory_allocated()
    reserved_b = torch.cuda.memory_reserved()
    print(
        f"{prefix} free={gb(free_b):.2f} GiB | "
        f"allocated={gb(alloc_b):.2f} GiB | "
        f"reserved={gb(reserved_b):.2f} GiB | "
        f"total={gb(total_b):.2f} GiB"
    )

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []


# -----------------------------
# Report expected sizes
# -----------------------------
input_shape = (batch_size, dim)
output_shape = input_shape

print("Expected tensor sizes:")
print(f"  input  {input_shape}:  {gb(tensor_nbytes(input_shape, dtype)):.2f} GiB")
print(f"  output {output_shape}: {gb(tensor_nbytes(output_shape, dtype)):.2f} GiB")
print()

assert torch.cuda.is_available(), "CUDA is not available"

torch.cuda.set_device(0)
cleanup()
print_mem("Before allocations:")

# -----------------------------
# Build model and input
# -----------------------------
model = Model().to(device=device)
x = get_inputs()[0].to(device=device, dtype=dtype)

torch.cuda.synchronize()
print_mem("After model+input:")

# -----------------------------
# Try one forward pass
# -----------------------------
try:
    with torch.inference_mode():
        start = time.perf_counter()
        y = model(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

    print("\nForward succeeded.")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Output dtype: {y.dtype}")
    print(f"Elapsed time: {(end - start) * 1000:.2f} ms")
    print_mem("After forward:")
    print(
        f"Peak allocated during run: "
        f"{gb(torch.cuda.max_memory_allocated()):.2f} GiB"
    )

except torch.cuda.OutOfMemoryError as e:
    print("\nForward failed with CUDA OOM.")
    print(str(e))
    print_mem("At OOM:")