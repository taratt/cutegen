"""Correctness and timing smoke test for the PTX runtime."""

import statistics

import torch

from cutegen.ptx_runtime import PtxModule, u32


PTX_SOURCE = r"""
.version 8.0
.target sm_89
.address_size 64

.visible .entry vector_add_f32(
    .param .u64 vector_add_f32_a,
    .param .u64 vector_add_f32_b,
    .param .u64 vector_add_f32_out,
    .param .u32 vector_add_f32_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [vector_add_f32_a];
    ld.param.u64 %rd2, [vector_add_f32_b];
    ld.param.u64 %rd3, [vector_add_f32_out];
    ld.param.u32 %r1, [vector_add_f32_n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

DONE:
    ret;
}
"""


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the PTX smoke test.")

    device = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (8, 9):
        raise SystemExit(
            f"This test targets sm_89, but the selected GPU is sm_{major}{minor}."
        )

    size = 1 << 20
    block = 256
    grid = (size + block - 1) // block
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    output = torch.empty_like(a)
    module = PtxModule(PTX_SOURCE)
    stream = torch.cuda.current_stream(device)

    def launch() -> None:
        module.launch(
            "vector_add_f32",
            grid=grid,
            block=block,
            arguments=[a, b, output, u32(size)],
            stream=stream,
        )

    for _ in range(10):
        launch()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(output, a + b)

    elapsed_ms = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        launch()
        end.record(stream)
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    print("PTX correctness: PASS")
    print(f"PTX mean time: {statistics.mean(elapsed_ms):.6f} ms")
    print(f"PTX min time: {min(elapsed_ms):.6f} ms")


if __name__ == "__main__":
    main()
