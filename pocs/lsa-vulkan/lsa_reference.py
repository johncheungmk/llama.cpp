"""Bounded synthetic latent-index experiment. No model weights or learned maps."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

MAX_TOKENS = 16384
MAX_DIM = 256


def matrix(rows, name="matrix"):
    if not rows or len(rows) > MAX_TOKENS:
        raise ValueError(f"{name}: expected 1..{MAX_TOKENS} rows")
    width = len(rows[0])
    if not 1 <= width <= MAX_DIM:
        raise ValueError(f"{name}: expected 1..{MAX_DIM} columns")
    if any(len(row) != width or any(not math.isfinite(x) for x in row) for row in rows):
        raise ValueError(f"{name}: ragged or non-finite input")
    return width


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("dot: dimensions differ")
    return math.fsum(x * y for x, y in zip(a, b))


def project(keys, projection):
    width = matrix(keys, "keys")
    if matrix(projection, "projection") != width:
        raise ValueError("projection: input dimension differs")
    return [[dot(key, row) for row in projection] for key in keys]


def attention(keys, values, query, ids):
    width = matrix(keys, "keys")
    value_width = matrix(values, "values")
    if len(values) != len(keys) or len(query) != width or not all(math.isfinite(x) for x in query):
        raise ValueError("attention: incompatible inputs")
    if not ids or len(ids) != len(set(ids)) or any(type(i) is not int or not 0 <= i < len(keys) for i in ids):
        raise ValueError("attention: invalid key indices")
    scores = [dot(keys[i], query) / math.sqrt(width) for i in ids]
    maximum = max(scores)
    probabilities = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(probabilities)
    return [math.fsum(p * values[i][j] for p, i in zip(probabilities, ids)) / denominator
            for j in range(value_width)]


@dataclass
class Node:
    lower: list[float]
    upper: list[float]
    minimum_id: int
    children: tuple["Node", ...] = ()
    ids: tuple[int, ...] = ()


class LatentTree:
    """Median-split hierarchy with bounded best-first leaf search on the CPU."""

    def __init__(self, keys, leaf_size=32):
        self.width = matrix(keys, "latent keys")
        if not 1 <= leaf_size <= 256:
            raise ValueError("leaf_size must be 1..256")
        self.keys = [list(row) for row in keys]
        self.leaf_size = leaf_size
        self.root = self._build(tuple(range(len(keys))))

    def _build(self, ids):
        count = len(ids)
        centroid = [math.fsum(self.keys[i][j] for i in ids) / count for j in range(self.width)]
        lower = [min(self.keys[i][j] for i in ids) for j in range(self.width)]
        upper = [max(self.keys[i][j] for i in ids) for j in range(self.width)]
        if count <= self.leaf_size:
            return Node(lower, upper, min(ids), ids=ids)
        axis = max(range(self.width), key=lambda j: math.fsum((self.keys[i][j] - centroid[j]) ** 2 for i in ids))
        ordered = tuple(sorted(ids, key=lambda i: (self.keys[i][axis], i)))
        middle = count // 2
        children = (self._build(ordered[:middle]), self._build(ordered[middle:]))
        return Node(lower, upper, min(ids), children=children)

    def select(self, query, k, candidate_budget, visible=None, sink=0, window=0):
        visible = len(self.keys) if visible is None else visible
        if type(visible) is not int or not 1 <= visible <= len(self.keys):
            raise ValueError("visible must name a nonempty causal prefix")
        if len(query) != self.width or not all(math.isfinite(x) for x in query):
            raise ValueError("latent query is invalid")
        if type(k) is not int or type(candidate_budget) is not int or not 1 <= k <= candidate_budget <= MAX_TOKENS:
            raise ValueError("expected 1 <= k <= candidate_budget <= MAX_TOKENS")
        if not 0 <= sink <= visible or not 0 <= window <= visible:
            raise ValueError("sink/window outside visible prefix")
        visited = 0
        if candidate_budget >= visible:
            candidates = list(range(visible))
        else:
            sequence = 0
            frontier = []

            def push(node):
                nonlocal sequence, visited
                if node.minimum_id >= visible:
                    return
                bound = math.fsum(q * (hi if q >= 0 else lo)
                                  for q, lo, hi in zip(query, node.lower, node.upper))
                heapq.heappush(frontier, (-bound, sequence, node))
                sequence += 1
                visited += 1

            push(self.root)
            candidates = []
            while frontier and len(candidates) < candidate_budget:
                _, _, node = heapq.heappop(frontier)
                if node.children:
                    for child in node.children:
                        push(child)
                else:
                    candidates.extend(i for i in node.ids if i < visible)
            candidates = candidates[:candidate_budget]
        ranked = sorted(candidates, key=lambda i: (-dot(query, self.keys[i]), i))[:k]
        required = set(range(sink)) | set(range(visible - window, visible))
        selected = sorted(set(ranked) | required)
        return selected, {"centroids_scored": visited, "keys_scored": len(candidates),
                          "selected_keys": len(selected), "visible_keys": visible}
