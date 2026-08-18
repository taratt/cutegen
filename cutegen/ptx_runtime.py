"""Minimal CUDA Driver API runtime for LLM-generated PTX kernels."""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Sequence
from typing import Union

import torch


ScalarArgument = Union[
    ctypes.c_byte,
    ctypes.c_ubyte,
    ctypes.c_short,
    ctypes.c_ushort,
    ctypes.c_int,
    ctypes.c_uint,
    ctypes.c_long,
    ctypes.c_ulong,
    ctypes.c_longlong,
    ctypes.c_ulonglong,
    ctypes.c_float,
    ctypes.c_double,
]
Dimension = Union[int, Sequence[int]]

_CU_JIT_INFO_LOG_BUFFER = 3
_CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
_CU_JIT_ERROR_LOG_BUFFER = 5
_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
_JIT_LOG_BUFFER_SIZE = 64 * 1024


def i32(value: int) -> ctypes.c_int32:
    return ctypes.c_int32(value)


def u32(value: int) -> ctypes.c_uint32:
    return ctypes.c_uint32(value)


def i64(value: int) -> ctypes.c_int64:
    return ctypes.c_int64(value)


def u64(value: int) -> ctypes.c_uint64:
    return ctypes.c_uint64(value)


def f32(value: float) -> ctypes.c_float:
    return ctypes.c_float(value)


def f64(value: float) -> ctypes.c_double:
    return ctypes.c_double(value)


def _as_dim3(value: Dimension) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, 1, 1)
    else:
        values = tuple(int(component) for component in value)
        if not 1 <= len(values) <= 3:
            raise ValueError(f"CUDA dimensions need 1-3 components, got {values}")
        result = values + (1,) * (3 - len(values))
    if any(component <= 0 for component in result):
        raise ValueError(f"CUDA dimensions must be positive, got {result}")
    return result


class _CudaDriver:
    def __init__(self) -> None:
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:
            raise RuntimeError(
                "Could not load libcuda.so.1; an NVIDIA driver is required for PTX"
            ) from exc

        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int

        self.lib.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.lib.cuModuleLoadData.restype = ctypes.c_int

        self.lib.cuModuleLoadDataEx.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.cuModuleLoadDataEx.restype = ctypes.c_int

        self.lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.lib.cuModuleGetFunction.restype = ctypes.c_int

        self.lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.cuLaunchKernel.restype = ctypes.c_int

        self.lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        self.lib.cuModuleUnload.restype = ctypes.c_int

        self.lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cuCtxGetCurrent.restype = ctypes.c_int

        self.lib.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.cuGetErrorName.restype = ctypes.c_int

        self.lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.cuGetErrorString.restype = ctypes.c_int

        self.check(self.lib.cuInit(0), "cuInit")

    def error_message(self, result: int, operation: str) -> str:
        name = ctypes.c_char_p()
        message = ctypes.c_char_p()
        self.lib.cuGetErrorName(result, ctypes.byref(name))
        self.lib.cuGetErrorString(result, ctypes.byref(message))
        error_name = name.value.decode() if name.value else f"CUDA_ERROR_{result}"
        error_message = message.value.decode() if message.value else "unknown error"
        return f"{operation} failed: {error_name}: {error_message}"

    def check(self, result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(self.error_message(result, operation))

    def load_ptx_module(
        self,
        module: ctypes.c_void_p,
        ptx_buffer: ctypes.Array,
    ) -> None:
        info_log = ctypes.create_string_buffer(_JIT_LOG_BUFFER_SIZE)
        error_log = ctypes.create_string_buffer(_JIT_LOG_BUFFER_SIZE)
        options = (ctypes.c_int * 4)(
            _CU_JIT_INFO_LOG_BUFFER,
            _CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            _CU_JIT_ERROR_LOG_BUFFER,
            _CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        )
        option_values = (ctypes.c_void_p * 4)(
            ctypes.cast(info_log, ctypes.c_void_p),
            ctypes.c_void_p(_JIT_LOG_BUFFER_SIZE),
            ctypes.cast(error_log, ctypes.c_void_p),
            ctypes.c_void_p(_JIT_LOG_BUFFER_SIZE),
        )
        result = self.lib.cuModuleLoadDataEx(
            ctypes.byref(module),
            ctypes.cast(ptx_buffer, ctypes.c_void_p),
            len(options),
            options,
            option_values,
        )
        if result == 0:
            return

        diagnostics = []
        error_text = error_log.value.decode("utf-8", errors="replace").strip()
        info_text = info_log.value.decode("utf-8", errors="replace").strip()
        if error_text:
            diagnostics.append(f"PTX JIT error log:\n{error_text}")
        if info_text:
            diagnostics.append(f"PTX JIT info log:\n{info_text}")
        diagnostic_suffix = (
            "\n" + "\n".join(diagnostics)
            if diagnostics
            else "\nPTX JIT produced no diagnostic log."
        )
        raise RuntimeError(
            self.error_message(result, "cuModuleLoadDataEx") + diagnostic_suffix
        )


_DRIVER: _CudaDriver | None = None
_DRIVER_LOCK = threading.Lock()


def _driver() -> _CudaDriver:
    global _DRIVER
    with _DRIVER_LOCK:
        if _DRIVER is None:
            _DRIVER = _CudaDriver()
        return _DRIVER


class PtxModule:
    """JIT-load PTX and launch its entry points on PyTorch CUDA streams."""

    def __init__(self, ptx: str) -> None:
        if not isinstance(ptx, str) or ".entry" not in ptx:
            raise ValueError("ptx must be a string containing at least one .entry")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        torch.cuda.init()
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        # PyTorch initializes the runtime lazily. A real allocation guarantees
        # that its primary context is current before the Driver API loads PTX.
        self._context_anchor = torch.empty(
            1,
            dtype=torch.uint8,
            device=torch.device("cuda", device),
        )
        self._driver = _driver()
        current_context = ctypes.c_void_p()
        self._driver.check(
            self._driver.lib.cuCtxGetCurrent(ctypes.byref(current_context)),
            "cuCtxGetCurrent",
        )
        if not current_context.value:
            raise RuntimeError("PyTorch did not establish a current CUDA context")
        self._module = ctypes.c_void_p()
        ptx_bytes = ptx.encode("utf-8") + b"\0"
        self._ptx_buffer = ctypes.create_string_buffer(ptx_bytes)
        self._driver.load_ptx_module(self._module, self._ptx_buffer)
        self._functions: dict[str, ctypes.c_void_p] = {}

    def function(self, name: str) -> ctypes.c_void_p:
        if name not in self._functions:
            function = ctypes.c_void_p()
            self._driver.check(
                self._driver.lib.cuModuleGetFunction(
                    ctypes.byref(function),
                    self._module,
                    name.encode("utf-8"),
                ),
                f"cuModuleGetFunction({name})",
            )
            self._functions[name] = function
        return self._functions[name]

    @staticmethod
    def _pack_argument(argument) -> ScalarArgument:
        if isinstance(argument, torch.Tensor):
            if not argument.is_cuda:
                raise ValueError("PTX tensor arguments must be CUDA tensors")
            return ctypes.c_uint64(argument.data_ptr())
        if isinstance(argument, ctypes._SimpleCData):
            return argument
        raise TypeError(
            "PTX arguments must be CUDA tensors or typed scalar helpers "
            "(i32/u32/i64/u64/f32/f64)"
        )

    def launch(
        self,
        name: str,
        grid: Dimension,
        block: Dimension,
        arguments: Sequence[object],
        *,
        shared_memory: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        grid_x, grid_y, grid_z = _as_dim3(grid)
        block_x, block_y, block_z = _as_dim3(block)
        packed = [self._pack_argument(argument) for argument in arguments]
        pointers = (ctypes.c_void_p * len(packed))(
            *[
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in packed
            ]
        )

        if stream is None:
            stream = torch.cuda.current_stream()
        stream_handle = ctypes.c_void_p(stream.cuda_stream)

        self._driver.check(
            self._driver.lib.cuLaunchKernel(
                self.function(name),
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                int(shared_memory),
                stream_handle,
                pointers,
                None,
            ),
            f"cuLaunchKernel({name})",
        )

    def close(self) -> None:
        module = getattr(self, "_module", None)
        if module and module.value:
            self._driver.check(
                self._driver.lib.cuModuleUnload(module),
                "cuModuleUnload",
            )
            module.value = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
