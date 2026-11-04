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


def _half(value, flush=False):
    rounded = np.asarray(value, np.float16).astype(np.float64)
    if not np.isfinite(rounded).all():
        raise ValueError("FP16 operand reference overflow")
    return np.where(np.abs(rounded) < 2**-14, 0, rounded) if flush else rounded


def _oracles(left, right, operation, mixed):
    yield "F32_operands", operation(left, right)
    if mixed:
        for flush in (False, True):
            a, b = _half(left, flush), _half(right, flush)
            suffix = "_flush_subnormals" if flush else ""
            yield "F16_left" + suffix, operation(a, right)
            yield "F16_right" + suffix, operation(left, b)
            yield "F16_both" + suffix, operation(a, b)


def _check_oracles(left, right, actual, operation, absolute_products, width, mixed):
    # F32 accumulator/reduction bound, separate from operand conversion error.
    unit = 2**-24
    gamma = (2 * width + 32) * unit / (1 - (2 * width + 32) * unit)
    tolerance = 1e-6 + gamma * absolute_products * 1.01
    checks = []
    for mode, expected in _oracles(left, right, operation, mixed):
        error = np.abs(expected - actual)
        checks.append((float(np.max(error / tolerance)), mode, float(np.max(error))))
    ratio, mode, error = min(checks)
    if not np.isfinite(ratio) or ratio > 1:
        raise RuntimeError(f"native result matches no allowed F32-accumulation operand oracle: ratio={ratio}, mode={mode}, error={error}")
    return {"operand_oracle": mode, "operand_oracle_max_abs_error": error,
            "operand_oracle_max_error_to_bound_ratio": ratio}


def verify_projection(values, projection, actual, mixed_operands=False):
    values = np.asarray(values, np.float64).reshape(-1, 128)
    projection = np.asarray(projection, np.float64)
    actual = np.asarray(actual, np.float64).reshape(len(values), len(projection))
    report = _check_oracles(values, projection, actual, lambda a, b: a @ b.T,
                            np.abs(values) @ np.abs(projection).T, 128, mixed_operands)
    report["full_F32_reference_max_abs_error"] = float(np.max(np.abs(values @ projection.T - actual)))
    return report


def verify(queries, keys, native, visible_mask, half_tile=False, mixed_operands=False):
    expected = scores(queries, keys)
    difference = np.abs(expected[visible_mask] - native[visible_mask])
    mixed_report = None
    if mixed_operands:
        q, k = np.asarray(queries, np.float64), np.asarray(keys, np.float64)
        # Select one consistent operand arithmetic for the whole captured layer.
        products = (np.abs(q).sum(axis=1) @ np.abs(k).T) / math.sqrt(128)
        mixed_report = _check_oracles(q, k, native[visible_mask],
                                     lambda a, b: scores(a, b)[visible_mask], products[visible_mask], q.shape[2], True)
        # Also bound the difference from unrounded mathematical scores.
        bound = fp16_tile_error_bound(queries, keys)
    else:
        bound = fp16_tile_error_bound(queries, keys) if half_tile else 1e-3 + 1e-4 * np.abs(expected)
    allowed = bound[visible_mask]
    if not np.isfinite(difference).all() or np.any(difference > allowed):
        raise RuntimeError(f"native QSA score exceeds its arithmetic error bound: {float(np.max(difference))}")
    result = {"max_abs_error": float(np.max(difference)), "max_allowed_error_bound": float(np.max(allowed)),
            "max_error_to_bound_ratio": float(np.max(difference / allowed)),
            "arithmetic": "stock_Vulkan_f16_dot4_with_f32_accumulation" if half_tile else "F32_accumulation_with_verified_operand_rounding" if mixed_operands else "F32"}
    if mixed_report:
        result.update(mixed_report)
    return result
