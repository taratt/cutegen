from skxoss.config import LLM_CONFIG_CODEGEN, SKXOSS_BASE_PATH
from skxoss.llm_api import create_llm_server_from_config
from skxoss.util import extract_first_code, read_file, debug_print
from skxoss.node import Node
from skxoss.code_editor import code_edit_apply_patches
from skxoss.util import extract_first_code, read_file, get_numbered_lines, src_to_lines, lines_to_src, debug_print
import json
import random

def codegen_initial(node: Node, addendum="", mode="original"):
    if mode == "original":
        return codegen_initial_original(node, addendum)
    else:
        raise ValueError(f"Invalid codegen_initial mode: {mode}")

def codegen_initial_original(node: Node, addendum=""):
    prompt = addendum
    prompt += """
You write custom CUDA kernels using CUTLASS to replace the pytorch operators in the given architecture to get speedups.

Here's an example to show you the syntax of inline embedding custom CUTLASS/CUDA operators in torch: The example given architecture is:
``` 
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
``` 

The example new arch with custom CUTLASS/CUDA kernels looks like this: 
```
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

atb_cpp_decl = r\"\"\"
#include <torch/extension.h>
torch::Tensor atb_gemm(torch::Tensor A, torch::Tensor B);
\"\"\"

atb_cuda_src = r\"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/layout.h>

// Types & layouts (fp32 baseline)
using ElementInputA      = float;
using ElementInputB      = float;
using ElementOutput      = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::ColumnMajor; // reinterpret A(KxM row-major) as (MxK col-major) → A^T
using LayoutB = cutlass::layout::RowMajor;    // B(KxN)
using LayoutC = cutlass::layout::RowMajor;    // C(MxN)

using GemmOp = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutA,
    ElementInputB, LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator
>;

torch::Tensor atb_gemm(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.dim()==2 && B.dim()==2,  "A and B must be 2D");
  TORCH_CHECK(A.scalar_type()==at::kFloat && B.scalar_type()==at::kFloat,
              "This build expects float32 tensors");

  // A:[K,M], B:[K,N]  → compute C = A^T @ B without explicit transpose
  int K = (int)A.size(0);
  int M = (int)A.size(1);
  TORCH_CHECK((int)B.size(0)==K, "B.size(0) must equal K");
  int N = (int)B.size(1);

  auto A_c = A.contiguous();
  auto B_c = B.contiguous();
  auto C   = torch::empty({M, N}, A.options());

  // Leading dimensions by layout
  int ldA = M; // ColumnMajor rows for logical (M,K)
  int ldB = N; // RowMajor cols
  int ldC = N; // RowMajor cols

  auto ptrA = reinterpret_cast<ElementInputA const*>(A_c.data_ptr<float>());
  auto ptrB = reinterpret_cast<ElementInputB const*>(B_c.data_ptr<float>());
  auto ptrC = reinterpret_cast<ElementOutput*      >(C.data_ptr<float>());

  GemmOp gemm;
  typename GemmOp::Arguments args(
      {M, N, K},
      {ptrA, ldA},
      {ptrB, ldB},
      {ptrC, ldC},
      {ptrC, ldC},
      {ElementAccumulator(1.0f), ElementAccumulator(0.0f)} // C = A^T * B
  );

  auto st = gemm.can_implement(args);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS: config not supported on this GPU");

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  st = gemm.initialize(args, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS: initialize() failed");

  st = gemm(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS: run failed");

  return C;
}
\"\"\"

cutlass_atb = load_inline(
    name="cutlass_atb_inline",
    cpp_sources=atb_cpp_decl,               # forward declaration (wrapper sees the symbol)
    cuda_sources=atb_cuda_src,              # implementation lives here
    functions=["atb_gemm"],                 # auto-generate pybind wrapper
    # extra_include_paths=[], NO NEED TO INCLUDE EXTRA BECAUSE CUTLASS HEADERS IN PATH ALREADY
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],              # DO NOT ADD gencode, JUST LET COMPILER TO FIND AVAILABLE ARCH, BECAUSE THIS LATER WILL BE TESTED
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._ext = cutlass_atb

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self._ext.atb_gemm(A, B)
```

Here's the architecture you need to optimize:
```
<REF>
```
Implement the architecture named Model with custom CUTLASS operators! Name your output architecture ModelNew, and make sure that it uses CUTLASS/CUDA kernels that you wrote for all of the underlying computation in forward(). Make sure that the written CUTLASS/CUDA is correct and complete, and that it implements the requested functions. Output the new code in CODEBLOCKS (wrap in ``` and ```). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! This is very important! Generate ONLY ONE kernel for the entire torch module, not one kernel per operator.
"""
    prompt += addendum
    prompt = prompt.replace("<REF>", node.ref)
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    src = llm_server(prompt)
    src = extract_first_code(src)
    return src

def codegen_optimize_original(node: Node, addendum=""):
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    prompt = f"""
You're given the following CUTLASS/CUDA source code:
```
{node.prev_src}
```
Optimized this code.!!
Please keep the given source code structure the same, including the function names, arguments, and return types. Please generate real code, NOT pseudocode, make sure the code will compile and is fully functional. Just output the new CUTLASS/CUDA + torch glue code, no other text, and NO testing code!.  Follow the original structure of the code (The optimized output architecture is named ModelNew with custom CUTLASS/CUDA kernel(s)). 
Do not use pre-optimized libraries like cuBLAS, do not change the precision, and do not use or fall back to pytorch operators, even for special cases! It is acceptable to specialize the code to the listed dimensions; it will only have to work for the exact shape listed. Output the entire new CUTLASS/CUDA kernel and the torch glue code. Output the new code in CODEBLOCKS (wrap in ``` and ```); the code MUST implement the functionalities of the previous version with new optimization ideas applied, this is very important! {addendum}
"""
    src = llm_server(prompt)
    src = extract_first_code(src)
    return src


def codegen_edits(node: Node, addendum=""):
    src_lines = src_to_lines(str(node.prev_src))
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    prompt = f"""
You're given the following CUTLASS/CUDA source code:
```
{get_numbered_lines(src_lines)}
```
Optimized this code.!!
"""
    prompt += "Please keep the given source code structure the same, including the function names, arguments, and return types. Please generate real code, NOT pseudocode, make sure the code will compile and is fully functional. ollow the original structure of the code (The output architecture is named ModelNew with custom CUTLASS/CUDA kernel(s)). Do not use pre-optimized libraries like cuBLAS, do not change the precision, and do not use or fall back to pytorch operators, even for special cases! It is acceptable to specialize the code to the listed dimensions; it will only have to work for the exact shape listed. The code MUST implement the functionalities of the previous version with new optimization ideas applied, this is very important!"
    prompt += read_file(SKXOSS_BASE_PATH + "/cutgen/prompts/code_editor_prompt.md")
    prompt += "Starting from the code example, output your reasoning for the edits in <reasoning></reasoning> tags. Output the edits in a codeblock ```json and ```."
    prompt += addendum
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    llm_response = llm_server(prompt)
    debug_print(f"node {node.uuid} codegen_edits llm_response: {llm_response}")
    src = ""
    # Apply code editor patches
    edits = extract_first_code(llm_response, code_language_types=["json", "python", ""])
    edits = json.loads(edits)
    edits = edits if isinstance(edits, list) else [edits]
    debug_print(edits)
    src_lines = code_edit_apply_patches(src_lines, edits)
    src = lines_to_src(src_lines)  
    return src

def codegen_optimize(node: Node, addendum="", mode="original"):
    if mode == "original":
        return codegen_optimize_original(node, addendum)
    elif mode == "edits":
        return codegen_edits(node, addendum)
    else:
        raise ValueError(f"Invalid codegen_optimize mode: {mode}")
