"""Small NumPy-only teacher-QSA distillation on orthonormal projections.

This is an independent experiment, not Prefrontal's D-LSA training code.
"""
import math

import numpy as np


def _probabilities(scores, visible, temperature):
    mask = np.arange(scores.shape[1])[None, :] < np.asarray(visible)[:, None]
    scaled = np.where(mask, scores / temperature, -1e30)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    logs = scaled - np.log(exp.sum(axis=1, keepdims=True))
    return probabilities, logs


def teacher_probabilities(keys, queries, visible, temperature=2.0):
    scores = np.maximum(np.einsum("thd,nd->thn", queries, keys, optimize=True), 0).sum(axis=1) / math.sqrt(128)
    return _probabilities(scores, visible, temperature)[0]


def objective(keys, queries, visible, projection, teacher=None, temperature=2.0, gradient=True):
    keys, queries, projection = (np.asarray(x, dtype=np.float64) for x in (keys, queries, projection))
    if teacher is None:
        teacher = teacher_probabilities(keys, queries, visible, temperature)
    q, k = queries @ projection.T, keys @ projection.T
    heads = np.einsum("thr,nr->thn", q, k, optimize=True)
    scores = np.maximum(heads, 0).sum(axis=1) / math.sqrt(128)
    probabilities, logs = _probabilities(scores, visible, temperature)
    loss = float(np.sum(teacher * (np.log(np.maximum(teacher, 1e-300)) - logs)) / len(queries))
    if not gradient:
        return loss, None
    ds = ((probabilities - teacher) / (len(queries) * temperature))[:, None, :] * (heads > 0) / math.sqrt(128)
    dq = np.einsum("thn,nr->thr", ds, k, optimize=True)
    dk = np.einsum("thn,thr->nr", ds, q, optimize=True)
    result = dq.reshape(-1, len(projection)).T @ queries.reshape(-1, 128) + dk.T @ keys
    return loss, result


def fit(keys, queries, visible, initial, steps=50, learning_rate=0.5, temperature=2.0, native_teacher_scores=None):
    keys, queries, projection = (np.asarray(x, dtype=np.float64) for x in (keys, queries, initial))
    visible = np.asarray(visible)
    if keys.ndim != 2 or keys.shape[1] != 128 or not 1 <= len(keys) <= 8192:
        raise ValueError("distillation supports at most 8192 pooled 128-wide keys")
    if queries.ndim != 3 or queries.shape[1:] != (4, 128) or not 1 <= len(queries) <= 128:
        raise ValueError("distillation supports 1..128 four-head queries")
    if visible.dtype.kind not in "iu" or visible.shape != (len(queries),) or np.any(visible < 1) or np.any(visible > len(keys)):
        raise ValueError("invalid distillation visibility")
    if projection.ndim != 2 or projection.shape[1] != 128 or not 1 <= len(projection) <= 128:
        raise ValueError("invalid distillation projection")
    if not all(np.isfinite(x).all() for x in (keys, queries, projection)) or not np.allclose(projection @ projection.T, np.eye(len(projection)), atol=1e-4):
        raise ValueError("distillation requires finite inputs and orthonormal rows")
    if type(steps) is not int or not 1 <= steps <= 100 or not 0 < learning_rate <= 1 or not 0 < temperature <= 10:
        raise ValueError("invalid bounded distillation settings")
    if native_teacher_scores is not None:
        native_teacher_scores = np.asarray(native_teacher_scores, dtype=np.float64)
        if native_teacher_scores.shape != (len(queries), len(keys)) or not np.isfinite(native_teacher_scores).all() or np.any(native_teacher_scores < 0):
            raise ValueError("invalid native teacher scores")
        teacher = _probabilities(native_teacher_scores, visible, temperature)[0]
    else:
        teacher = teacher_probabilities(keys, queries, visible, temperature)
    trace = [objective(keys, queries, visible, projection, teacher, temperature, False)[0]]
    for _ in range(steps):
        loss, grad = objective(keys, queries, visible, projection, teacher, temperature)
        symmetric = (grad @ projection.T + projection @ grad.T) * 0.5
        tangent = grad - symmetric @ projection
        norm = float(np.linalg.norm(tangent))
        if not np.isfinite(loss) or not np.isfinite(norm):
            raise ArithmeticError("non-finite distillation objective")
        if loss < 1e-12 or norm < 1e-10:
            break
        # KL is averaged over queries and can have a very small gradient. Set a
        # bounded tangent displacement, then let training-only backtracking shrink it.
        step = learning_rate / norm
        accepted = False
        for _ in range(12):
            candidate = np.linalg.qr((projection - step * tangent).T, mode="reduced")[0].T
            new_loss = objective(keys, queries, visible, candidate, teacher, temperature, False)[0]
            if np.isfinite(new_loss) and new_loss < loss - 1e-12:
                projection = candidate
                trace.append(new_loss)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return projection.astype(np.float32), {
        "method": "independent_orthonormal_QSA_softmax_teacher_distillation",
        "temperature": temperature, "learning_rate": learning_rate,
        "step_rule": "unit_tangent_direction_with_training_only_backtracking",
        "requested_steps": steps, "accepted_steps": len(trace) - 1,
        "training_kl_trace": trace, "heldout_used_for_training": False,
        "native_teacher_scores_used": native_teacher_scores is not None,
    }
