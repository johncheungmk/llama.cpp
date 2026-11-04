"""References for the pinned CPU and Vulkan QSA indexers.

Vulkan lightning_indexer.comp stages Q/K in f16vec4 and converts each four-term
dot result to float before accumulation. Its conservative error bound includes
operand rounding, seven half operations per dot4, and float accumulation.
"""
import math

import numpy as np


def scores(queries, keys):
    q, k = np.asarray(queries, np.float64), np.asarray(keys, np.float64)
    return np.maximum(np.einsum("thd,nd->thn", q, k, optimize=True), 0).sum(axis=1) / math.sqrt(128)


def fp16_tile_error_bound(queries, keys):
    q, k = np.asarray(queries, np.float64), np.asarray(keys, np.float64)
    q16, k16 = q.astype(np.float16).astype(np.float64), k.astype(np.float16).astype(np.float64)
    if not np.isfinite(q16).all() or not np.isfinite(k16).all() or 4 * np.max(np.abs(q16)) * np.max(np.abs(k16)) >= 60000:
        raise ValueError("half-dot bound requires finite, non-overflowing dot4 inputs")
    # Cover implementations that flush half subnormals, rather than assuming
    # relative-error bounds apply to underflowing inputs/intermediates.
    normal_min = 2**-14
    dq = np.abs(q - q16) + np.where(np.abs(q16) < normal_min, np.abs(q16), 0)
    dk = np.abs(k - k16) + np.where(np.abs(k16) < normal_min, np.abs(k16), 0)
    operand = dq.sum(axis=1) @ np.abs(k).T + np.abs(q16).sum(axis=1) @ dk.T
    magnitudes = np.abs(q16).sum(axis=1) @ np.abs(k16).T
    half_unit, float_unit = 2**-11, 2**-24
    gamma_half = 7 * half_unit / (1 - 7 * half_unit)
    gamma_float = 40 * float_unit / (1 - 40 * float_unit)
    underflow = 7 * (q.shape[2] // 4) * q.shape[1] * normal_min
    return (operand + (gamma_half + gamma_float) * magnitudes + underflow) / math.sqrt(128) + 1e-6


def verify(queries, keys, native, visible_mask, half_tile=False):
    expected = scores(queries, keys)
    difference = np.abs(expected[visible_mask] - native[visible_mask])
    bound = fp16_tile_error_bound(queries, keys) if half_tile else 1e-3 + 1e-4 * np.abs(expected)
    allowed = bound[visible_mask]
    if not np.isfinite(difference).all() or np.any(difference > allowed):
        raise RuntimeError(f"native QSA score exceeds its arithmetic error bound: {float(np.max(difference))}")
    return {"max_abs_error": float(np.max(difference)), "max_allowed_error_bound": float(np.max(allowed)),
            "max_error_to_bound_ratio": float(np.max(difference / allowed)),
            "arithmetic": "stock_Vulkan_f16_dot4_with_f32_accumulation" if half_tile else "F32"}
