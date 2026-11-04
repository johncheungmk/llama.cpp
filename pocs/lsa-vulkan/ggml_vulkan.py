"""Small ctypes client for the public ggml C API at llama.cpp b11681."""
from __future__ import annotations

import array
import ctypes as C
import hashlib
import math
import os
from pathlib import Path
import time

from lsa_reference import matrix

P = C.c_void_p
SIZE = C.c_size_t
I64 = C.c_int64
F32 = 0
I32 = 26
F32_PRECISION = 10
MAX_BUFFER_BYTES = 64 * 1024 * 1024


class InitParams(C.Structure):
    _fields_ = [("mem_size", SIZE), ("mem_buffer", P), ("no_alloc", C.c_bool)]


class Vulkan:
    def __init__(self, runtime):
        self.backend = None
        if os.name != "nt" or C.sizeof(P) != 8:
            raise RuntimeError("This ABI harness requires 64-bit Windows")
        # Keep the reference experiment out of implicit F16 cooperative-matrix paths.
        os.environ["GGML_VK_DISABLE_F16"] = "1"
        os.environ["GGML_VK_DISABLE_COOPMAT"] = "1"
        runtime = Path(runtime).resolve(strict=True)
        self.dll_directory = os.add_dll_directory(str(runtime))
        names = ["ggml-base.dll", "ggml.dll", "ggml-vulkan.dll"]
        self.libraries = [C.CDLL(str(runtime / name)) for name in names]
        self.fingerprints = {name: hashlib.sha256((runtime / name).read_bytes()).hexdigest() for name in names}
        self._bind()
        self.backend = self.vk_init(0)
        if not self.backend or not self.is_vk(self.backend):
            raise RuntimeError("A Vulkan backend is required; CPU fallback is refused")
        description = C.create_string_buffer(512)
        self.vk_description(0, description, len(description))
        self.device = description.value.decode("utf-8", errors="replace")

    def _function(self, name, result, arguments):
        for library in self.libraries:
            try:
                function = getattr(library, name)
                function.restype, function.argtypes = result, arguments
                return function
            except AttributeError:
                pass
        raise RuntimeError(f"Required b11681 C API export is absent: {name}")

    def _bind(self):
        definitions = {
            "vk_init": ("ggml_backend_vk_init", P, [SIZE]),
            "is_vk": ("ggml_backend_is_vk", C.c_bool, [P]),
            "vk_description": ("ggml_backend_vk_get_device_description", None, [C.c_int, P, SIZE]),
            "init": ("ggml_init", P, [InitParams]),
            "free_context": ("ggml_free", None, [P]),
            "free_backend": ("ggml_backend_free", None, [P]),
            "new_tensor": ("ggml_new_tensor_2d", P, [P, C.c_int, I64, I64]),
            "mul_mat": ("ggml_mul_mat", P, [P, P, P]),
            "scale": ("ggml_scale", P, [P, P, C.c_float]),
            "softmax": ("ggml_soft_max", P, [P, P]),
            "get_rows": ("ggml_get_rows", P, [P, P, P]),
            "transpose": ("ggml_transpose", P, [P, P]),
            "contiguous": ("ggml_cont", P, [P, P]),
            "set_precision": ("ggml_prec_set_acc", C.c_bool, [P, C.c_int]),
            "set_source_precision": ("ggml_prec_set_src", C.c_bool, [P, C.c_int, C.c_int]),
            "new_graph": ("ggml_new_graph_custom", P, [P, SIZE, C.c_bool]),
            "expand_graph": ("ggml_build_forward_expand", None, [P, P]),
            "allocate": ("ggml_backend_alloc_ctx_tensors", P, [P, P]),
            "buffer_size": ("ggml_backend_buffer_get_size", SIZE, [P]),
            "free_buffer": ("ggml_backend_buffer_free", None, [P]),
            "set_tensor": ("ggml_backend_tensor_set", None, [P, P, SIZE, SIZE]),
            "get_tensor": ("ggml_backend_tensor_get", None, [P, P, SIZE, SIZE]),
            "nbytes": ("ggml_nbytes", SIZE, [P]),
            "nelements": ("ggml_nelements", I64, [P]),
            "supports": ("ggml_backend_supports_op", C.c_bool, [P, P]),
            "compute": ("ggml_backend_graph_compute", C.c_int, [P, P]),
            "synchronize": ("ggml_backend_synchronize", None, [P]),
        }
        for attribute, (name, result, arguments) in definitions.items():
            setattr(self, attribute, self._function(name, result, arguments))

    def close(self):
        if self.backend:
            self.free_backend(self.backend)
            self.backend = None
        self.dll_directory.close()

    def graph(self):
        return Graph(self)

    def project(self, keys, projection):
        width = matrix(keys, "keys")
        if matrix(projection, "projection") != width:
            raise ValueError("projection dimensions differ")
        with self.graph() as graph:
            p = graph.input(projection)
            k = graph.input(keys)
            output = graph.matmul(p, k)
            graph.allocate(output)
            elapsed, flat = graph.run(output)
            rank = len(projection)
            return [flat[i:i + rank] for i in range(0, len(flat), rank)], elapsed

    def attention_graph(self, keys, values, query, ids=None):
        width = matrix(keys, "keys")
        matrix(values, "values")
        if len(keys) != len(values) or len(query) != width or not all(math.isfinite(x) for x in query):
            raise ValueError("attention dimensions differ")
        if ids is not None and (not ids or len(ids) != len(set(ids)) or any(type(i) is not int or not 0 <= i < len(keys) for i in ids)):
            raise ValueError("attention indices are invalid")
        graph = self.graph()
        try:
            k, v, q = graph.input(keys), graph.input(values), graph.input([query])
            if ids is not None:
                indices = graph.input([ids], integer=True)
                k, v = graph.op(self.get_rows, k, indices), graph.op(self.get_rows, v, indices)
            scores = graph.matmul(k, q)
            scaled = graph.op(self.scale, scores, 1.0 / math.sqrt(width))
            weights = graph.op(self.softmax, scaled)
            transposed = graph.op(self.contiguous, graph.op(self.transpose, v))
            output = graph.matmul(transposed, weights)
            graph.allocate(output)
            return graph, output
        except BaseException:
            graph.close()
            raise


class Graph:
    def __init__(self, engine):
        self.engine, self.buffer = engine, None
        self.context = engine.init(InitParams(2 * 1024 * 1024, None, True))
        if not self.context:
            raise MemoryError("ggml context allocation failed")
        self.inputs, self.operations = [], []

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def close(self):
        if self.buffer:
            self.engine.free_buffer(self.buffer)
            self.buffer = None
        if self.context:
            self.engine.free_context(self.context)
            self.context = None

    def input(self, rows, integer=False):
        if integer:
            if len(rows) != 1 or not rows[0] or len(rows[0]) > 16384:
                raise ValueError("indices outside bounded experiment")
            columns = len(rows[0])
        else:
            columns = matrix(rows)
        data = array.array("i" if integer else "f", (x for row in rows for x in row))
        if data.itemsize != 4:
            raise RuntimeError("32-bit array storage is required")
        n0, n1 = (columns, len(rows))
        # A 1-D index tensor is represented as [number of rows, 1].
        tensor = self.engine.new_tensor(self.context, I32 if integer else F32, n0, n1)
        if not tensor:
            raise MemoryError("input tensor allocation failed")
        self.inputs.append((tensor, data.tobytes()))
        return tensor

    def op(self, function, *arguments):
        tensor = function(self.context, *arguments)
        if not tensor or not self.engine.supports(self.engine.backend, tensor):
            raise RuntimeError("Vulkan does not support an operation; fallback refused")
        self.operations.append(tensor)
        return tensor

    def matmul(self, a, b):
        tensor = self.op(self.engine.mul_mat, a, b)
        if not self.engine.set_precision(tensor, F32_PRECISION):
            raise RuntimeError("Cannot require F32 accumulation")
        if not self.engine.set_source_precision(tensor, F32_PRECISION, 1):
            raise RuntimeError("Cannot require F32 source precision")
        return tensor

    def allocate(self, output):
        self.graph = self.engine.new_graph(self.context, 256, False)
        if not self.graph:
            raise MemoryError("graph allocation failed")
        self.engine.expand_graph(self.graph, output)
        self.buffer = self.engine.allocate(self.context, self.engine.backend)
        if not self.buffer:
            raise MemoryError("Vulkan buffer allocation failed")
        self.bytes = self.engine.buffer_size(self.buffer)
        if self.bytes > MAX_BUFFER_BYTES:
            raise MemoryError("Experiment exceeded its 64 MiB tensor-buffer limit")
        for tensor, data in self.inputs:
            if len(data) != self.engine.nbytes(tensor):
                raise RuntimeError("C API tensor size differs from input storage")
            storage = C.create_string_buffer(data)
            self.engine.set_tensor(tensor, storage, 0, len(data))

    def run(self, output):
        start = time.perf_counter()
        status = self.engine.compute(self.engine.backend, self.graph)
        self.engine.synchronize(self.engine.backend)
        elapsed = time.perf_counter() - start
        if status != 0:
            raise RuntimeError(f"Vulkan graph failed with ggml status {status}")
        count = self.engine.nelements(output)
        if not 1 <= count <= 16384 * 256 or self.engine.nbytes(output) != count * 4:
            raise RuntimeError("Unexpected output tensor storage")
        result = (C.c_float * count)()
        self.engine.get_tensor(output, result, 0, count * 4)
        values = list(result)
        if not all(math.isfinite(x) for x in values):
            raise ArithmeticError("Vulkan output contains non-finite values")
        return elapsed, values
