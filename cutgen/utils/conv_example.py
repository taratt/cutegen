#!/usr/bin/env python3
import torch


# ============================================================
# 1) Reference task code
# ============================================================
REF_SRC = r'''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = (3, 5, 7)
depth_in = 16
height_in = 32
width_in = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth_in, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
'''


# ============================================================
# 2) Generated source
#    This is the corrected version with explicit mY(oc, p) store.
# ============================================================
GEN_SRC = r'''
import os
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.cpp_extension import load_inline

cuda_decl = r"""
#include <torch/extension.h>
torch::Tensor convtrans3d_forward(torch::Tensor x,
                                  torch::Tensor w,
                                  c10::optional<torch::Tensor> b,
                                  int stride_d, int stride_h, int stride_w,
                                  int pad_d, int pad_h, int pad_w,
                                  int out_pad_d, int out_pad_h, int out_pad_w,
                                  int groups);
"""

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>

using namespace cute;

__device__ __forceinline__ int pos_mod(int a, int m) {
  int r = a % m;
  return r < 0 ? r + m : r;
}

template <typename T>
__global__ void convtrans3d_kernel(
    const T* __restrict__ X,
    const T* __restrict__ W,
    const T* __restrict__ B,
    T* __restrict__ Y,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups)
{
  constexpr int Threads    = 128;
  constexpr int PVec       = 1;
  constexpr int OcPerBlock = 4;
  constexpr int KChunk     = 64;

  const int tid = threadIdx.x;
  const int P_total = D_out * H_out * W_out;
  const int PTile   = Threads * PVec;

  const int Cg_in    = C_in / groups;
  const int Cg_out   = C_out / groups;
  const int Kspatial = Kd * Kh * Kw;
  const int K_total  = Cg_in * Kspatial;

  const int n = blockIdx.z;

  const int tiles_per_group = (Cg_out + OcPerBlock - 1) / OcPerBlock;
  const int g       = blockIdx.y / tiles_per_group;
  const int oc_tile = blockIdx.y - g * tiles_per_group;
  const int oc_base = g * Cg_out + oc_tile * OcPerBlock;

  const int64_t sXn = (int64_t)C_in * D_in * H_in * W_in;
  const int64_t sXc = (int64_t)D_in * H_in * W_in;

  const int64_t sYn = (int64_t)C_out * D_out * H_out * W_out;
  const int64_t sYc = (int64_t)D_out * H_out * W_out;

  auto mX = make_tensor(
      make_gmem_ptr(X + (int64_t)n * sXn),
      make_shape(C_in, D_in * H_in * W_in),
      make_stride(sXc, Int<1>{})
  );

  auto mY = make_tensor(
      make_gmem_ptr(Y + (int64_t)n * sYn),
      make_shape(C_out, P_total),
      make_stride(sYc, Int<1>{})
  );

  auto mW = make_tensor(
      make_gmem_ptr(W),
      make_shape(C_in, Cg_out, Kd, Kh, Kw),
      make_stride((int64_t)Cg_out * Kd * Kh * Kw,
                  (int64_t)Kd * Kh * Kw,
                  (int64_t)Kh * Kw,
                  (int64_t)Kw,
                  Int<1>{})
  );

  __shared__ T   sW_raw[OcPerBlock * KChunk];
  __shared__ int sIC_raw[KChunk];
  __shared__ int sKD_raw[KChunk];
  __shared__ int sKH_raw[KChunk];
  __shared__ int sKW_raw[KChunk];

  auto sW = make_tensor(
      make_smem_ptr(sW_raw),
      make_shape(Int<OcPerBlock>{}, Int<KChunk>{}),
      make_stride(Int<KChunk>{}, Int<1>{})
  );
  auto sIC = make_tensor(make_smem_ptr(sIC_raw), make_shape(Int<KChunk>{}), make_stride(Int<1>{}));
  auto sKD = make_tensor(make_smem_ptr(sKD_raw), make_shape(Int<KChunk>{}), make_stride(Int<1>{}));
  auto sKH = make_tensor(make_smem_ptr(sKH_raw), make_shape(Int<KChunk>{}), make_stride(Int<1>{}));
  auto sKW = make_tensor(make_smem_ptr(sKW_raw), make_shape(Int<KChunk>{}), make_stride(Int<1>{}));

  int p_idx[PVec];
  int zyx[3 * PVec];

  #pragma unroll
  for (int i = 0; i < PVec; ++i) {
    int p = blockIdx.x * PTile + tid + i * Threads;
    p_idx[i] = p;

    int z = 0, y = 0, x = 0;
    if (p < P_total) {
      int plane = H_out * W_out;
      z = p / plane;
      int rem = p - z * plane;
      y = rem / W_out;
      x = rem - y * W_out;
    }
    zyx[3 * i + 0] = z;
    zyx[3 * i + 1] = y;
    zyx[3 * i + 2] = x;
  }

  float acc[OcPerBlock][PVec];
  #pragma unroll
  for (int oc_i = 0; oc_i < OcPerBlock; ++oc_i) {
    int oc = oc_base + oc_i;
    float bval = (B != nullptr && oc < C_out) ? (float)B[oc] : 0.0f;
    #pragma unroll
    for (int i = 0; i < PVec; ++i) {
      acc[oc_i][i] = bval;
    }
  }

  for (int k0 = 0; k0 < K_total; k0 += KChunk) {
    for (int kk = tid; kk < KChunk; kk += Threads) {
      int kk_global = k0 + kk;

      int icg = -1, kd = -1, kh = -1, kw = -1;
      if (kk_global < K_total) {
        icg = kk_global / Kspatial;
        int t = kk_global - icg * Kspatial;
        kd = t / (Kh * Kw);
        int rem = t - kd * (Kh * Kw);
        kh = rem / Kw;
        kw = rem - kh * Kw;
      }

      sIC(kk) = icg;
      sKD(kk) = kd;
      sKH(kk) = kh;
      sKW(kk) = kw;

      #pragma unroll
      for (int oc_i = 0; oc_i < OcPerBlock; ++oc_i) {
        int oc = oc_base + oc_i;
        T wval = T(0);
        if (kk_global < K_total && oc < C_out) {
          int oc_rel = oc - g * Cg_out;
          int ic_global = g * Cg_in + icg;
          wval = mW(ic_global, oc_rel, kd, kh, kw);
        }
        sW(oc_i, kk) = wval;
      }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < PVec; ++i) {
      int p = p_idx[i];
      if (p >= P_total) continue;

      int z = zyx[3 * i + 0];
      int y = zyx[3 * i + 1];
      int x = zyx[3 * i + 2];

      #pragma unroll
      for (int kk = 0; kk < KChunk; ++kk) {
        int kk_global = k0 + kk;
        if (kk_global >= K_total) break;

        int icg = sIC(kk);
        int kd  = sKD(kk);
        int kh  = sKH(kk);
        int kw  = sKW(kk);

        if (icg < 0) continue;

        int zz = z + pad_d - kd;
        int yy = y + pad_h - kh;
        int xx = x + pad_w - kw;

        if (pos_mod(zz, stride_d) != 0) continue;
        if (pos_mod(yy, stride_h) != 0) continue;
        if (pos_mod(xx, stride_w) != 0) continue;

        int iz = zz / stride_d;
        int iy = yy / stride_h;
        int ix = xx / stride_w;

        if ((unsigned)iz >= (unsigned)D_in) continue;
        if ((unsigned)iy >= (unsigned)H_in) continue;
        if ((unsigned)ix >= (unsigned)W_in) continue;

        int ic_global = g * Cg_in + icg;
        int p_in = (iz * H_in + iy) * W_in + ix;
        float xval = (float)mX(ic_global, p_in);

        #pragma unroll
        for (int oc_i = 0; oc_i < OcPerBlock; ++oc_i) {
          int oc = oc_base + oc_i;
          if (oc >= C_out) continue;
          float wval = (float)sW(oc_i, kk);
          acc[oc_i][i] += xval * wval;
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int oc_i = 0; oc_i < OcPerBlock; ++oc_i) {
    int oc = oc_base + oc_i;
    if (oc >= C_out) continue;
    #pragma unroll
    for (int i = 0; i < PVec; ++i) {
      int p = p_idx[i];
      if (p >= P_total) continue;
      mY(oc, p) = static_cast<T>(acc[oc_i][i]);
    }
  }
}

torch::Tensor convtrans3d_forward(torch::Tensor x,
                                  torch::Tensor w,
                                  c10::optional<torch::Tensor> b,
                                  int stride_d, int stride_h, int stride_w,
                                  int pad_d, int pad_h, int pad_w,
                                  int out_pad_d, int out_pad_h, int out_pad_w,
                                  int groups) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(w.is_cuda(), "w must be CUDA");
  if (b.has_value()) TORCH_CHECK(b.value().is_cuda(), "bias must be CUDA if provided");

  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");
  if (b.has_value()) TORCH_CHECK(b.value().scalar_type() == at::kFloat, "bias must be float32");

  TORCH_CHECK(x.dim() == 5, "x must be [N, C_in, D, H, W]");
  TORCH_CHECK(w.dim() == 5, "w must be [C_in, C_out/groups, Kd, Kh, Kw]");

  int64_t N = x.size(0);
  int64_t C_in = x.size(1);
  int64_t D_in = x.size(2);
  int64_t H_in = x.size(3);
  int64_t W_in = x.size(4);

  int64_t Cin_w = w.size(0);
  int64_t Cout_per_g = w.size(1);
  int64_t Kd = w.size(2);
  int64_t Kh = w.size(3);
  int64_t Kw = w.size(4);

  TORCH_CHECK(Cin_w == C_in, "w.size(0) must equal C_in");
  TORCH_CHECK(groups > 0, "groups must be > 0");
  TORCH_CHECK((C_in % groups) == 0, "C_in must be divisible by groups");

  int64_t C_out = Cout_per_g * groups;

  int64_t D_out = (D_in - 1) * stride_d - 2 * pad_d + Kd + out_pad_d;
  int64_t H_out = (H_in - 1) * stride_h - 2 * pad_h + Kh + out_pad_h;
  int64_t W_out = (W_in - 1) * stride_w - 2 * pad_w + Kw + out_pad_w;

  TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Invalid output size");

  auto x_c = x.contiguous();
  auto w_c = w.contiguous();

  torch::Tensor b_c;
  const float* b_ptr = nullptr;
  if (b.has_value()) {
    b_c = b.value().contiguous();
    TORCH_CHECK(b_c.dim() == 1 && b_c.size(0) == C_out, "bias must be [C_out]");
    b_ptr = b_c.data_ptr<float>();
  }

  auto y = torch::empty({N, C_out, D_out, H_out, W_out}, x.options());

  constexpr int Threads = 128;
  constexpr int PVec = 1;
  constexpr int OcPerBlock = 4;
  const int64_t P_total = D_out * H_out * W_out;
  const int64_t PTile = Threads * PVec;
  const int64_t Cg_out = C_out / groups;
  const int64_t tiles_per_group = (Cg_out + OcPerBlock - 1) / OcPerBlock;

  dim3 block(Threads);
  dim3 grid((unsigned)((P_total + PTile - 1) / PTile),
            (unsigned)(groups * tiles_per_group),
            (unsigned)N);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  convtrans3d_kernel<float><<<grid, block, 0, stream>>>(
      x_c.data_ptr<float>(),
      w_c.data_ptr<float>(),
      b_ptr,
      y.data_ptr<float>(),
      (int)N, (int)C_in, (int)C_out,
      (int)D_in, (int)H_in, (int)W_in,
      (int)D_out, (int)H_out, (int)W_out,
      (int)Kd, (int)Kh, (int)Kw,
      stride_d, stride_h, stride_w,
      pad_d, pad_h, pad_w,
      out_pad_d, out_pad_h, out_pad_w,
      groups
  );

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "convtrans3d_kernel launch failed: ", cudaGetErrorString(err));

  return y;
}
"""

convtrans3d_ext = load_inline(
    name="cute_convtrans3d_ext_true_fixed",
    cpp_sources=cuda_decl,
    cuda_sources=cuda_src,
    functions=["convtrans3d_forward"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_include_paths=["/home/tarasaba/cutlass/include", "/home/tarasaba/cutlass/include/cute/"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple = (1, 1, 1),
                 padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0),
                 groups: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        assert len(kernel_size) == 3
        assert len(stride) == 3
        assert len(padding) == 3
        assert len(output_padding) == 3
        assert out_channels % groups == 0
        assert in_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.padding = tuple(padding)
        self.output_padding = tuple(output_padding)
        self.groups = groups

        Cout_per_g = out_channels // groups
        Kd, Kh, Kw = self.kernel_size
        self.weight = nn.Parameter(torch.empty(in_channels, Cout_per_g, Kd, Kh, Kw))

        if bias:
          self.bias = nn.Parameter(torch.empty(out_channels))
        else:
          self.register_parameter("bias", None)

        self.reset_parameters()
        self._ext = convtrans3d_ext

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        opd, oph, opw = self.output_padding
        return self._ext.convtrans3d_forward(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias if self.bias is not None else None,
            int(sd), int(sh), int(sw),
            int(pd), int(ph), int(pw),
            int(opd), int(oph), int(opw),
            int(self.groups),
        )
'''


# ============================================================
# 3) Helpers
# ============================================================
def load_ctx(src: str):
    ctx = {}
    exec(src, ctx, ctx)
    return ctx


def instantiate_model_with_inputs(ctx, class_name: str, init_inputs, device: torch.device):
    ModelCls = ctx[class_name]
    model = ModelCls(*init_inputs).to(device)
    model.eval()
    return model


def clone_weights_from_ref_to_gen(ref_model, gen_model):
    ref_sd = ref_model.state_dict()
    gen_sd = gen_model.state_dict()

    copied = []

    # First try exact-name matches
    for gk in list(gen_sd.keys()):
        if gk in ref_sd and tuple(gen_sd[gk].shape) == tuple(ref_sd[gk].shape):
            gen_sd[gk].copy_(ref_sd[gk])
            copied.append((gk, gk))

    # If exact names failed, fall back to common suffix-based mapping
    # e.g. conv_transpose3d.weight -> weight
    for gk in list(gen_sd.keys()):
        if any(gk == pair[0] for pair in copied):
            continue

        candidates = []
        for rk in ref_sd.keys():
            if rk.endswith("." + gk) or rk == gk:
                if tuple(ref_sd[rk].shape) == tuple(gen_sd[gk].shape):
                    candidates.append(rk)

        if len(candidates) == 1:
            rk = candidates[0]
            gen_sd[gk].copy_(ref_sd[rk])
            copied.append((gk, rk))

    gen_model.load_state_dict(gen_sd, strict=False)
    return copied


@torch.no_grad()
def check_correctness(device="cuda", atol=1e-4, rtol=1e-4, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ref_ctx = load_ctx(REF_SRC)
    gen_ctx = load_ctx(GEN_SRC)

    init_inputs = ref_ctx["get_init_inputs"]() if "get_init_inputs" in ref_ctx else []

    ref_model = instantiate_model_with_inputs(ref_ctx, "Model", init_inputs, torch.device(device))
    gen_model = instantiate_model_with_inputs(gen_ctx, "ModelNew", init_inputs, torch.device(device))

    copied = clone_weights_from_ref_to_gen(ref_model, gen_model)

    inputs = ref_ctx["get_inputs"]()
    inputs = [x.to(device) if torch.is_tensor(x) else x for x in inputs]

    y_ref = ref_model(*inputs)
    y_gen = gen_model(*inputs)

    same_shape = tuple(y_ref.shape) == tuple(y_gen.shape)
    ok = same_shape and torch.allclose(y_ref, y_gen, atol=atol, rtol=rtol)

    max_abs = (y_ref - y_gen).abs().max().item() if same_shape else float("inf")
    denom = y_ref.abs().max().item() if same_shape else 0.0
    max_rel = max_abs / (denom + 1e-12) if same_shape else float("inf")

    print("\n[correctness]")
    print("init_inputs:", init_inputs)
    print("copied_params:", copied)
    print("ref_shape:", tuple(y_ref.shape))
    print("gen_shape:", tuple(y_gen.shape))
    print("allclose:", ok)
    print("max_abs_err:", max_abs)
    print("max_rel_err:", max_rel)

    return ok


@torch.no_grad()
def benchmark_model(model, inputs, warmups=10, iters=50):
    for _ in range(warmups):
        _ = model(*inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    return mean_ms, std_ms


@torch.no_grad()
def check_performance(device="cuda", seed=0, warmups=10, iters=50):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ref_ctx = load_ctx(REF_SRC)
    gen_ctx = load_ctx(GEN_SRC)

    init_inputs = ref_ctx["get_init_inputs"]() if "get_init_inputs" in ref_ctx else []

    ref_model = instantiate_model_with_inputs(ref_ctx, "Model", init_inputs, torch.device(device))
    gen_model = instantiate_model_with_inputs(gen_ctx, "ModelNew", init_inputs, torch.device(device))

    clone_weights_from_ref_to_gen(ref_model, gen_model)

    inputs = ref_ctx["get_inputs"]()
    inputs = [x.to(device) if torch.is_tensor(x) else x for x in inputs]

    ref_mean, ref_std = benchmark_model(ref_model, inputs, warmups=warmups, iters=iters)
    gen_mean, gen_std = benchmark_model(gen_model, inputs, warmups=warmups, iters=iters)

    speedup = ref_mean / gen_mean if gen_mean > 0 else float("inf")

    print("\n[performance]")
    print(f"ref_mean_ms: {ref_mean:.6f}")
    print(f"ref_std_ms : {ref_std:.6f}")
    print(f"gen_mean_ms: {gen_mean:.6f}")
    print(f"gen_std_ms : {gen_std:.6f}")
    print(f"speedup    : {speedup:.4f}x")

    return speedup


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required"

    ok = check_correctness(
        device="cuda",
        atol=1e-4,
        rtol=1e-4,
        seed=0,
    )

    if ok:
        check_performance(
            device="cuda",
            seed=0,
            warmups=10,
            iters=50,
        )
    else:
        print("\nSkipping performance because correctness failed.")