"""Per-position gap between a spliced cache and an honest one.

A shadow track only needs to run while the two differ, so this is what a
stopping rule would be built on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scm.rotate import relative_l2
from scm.splice import Layer


@dataclass(frozen=True, slots=True)
class Decay:
    """Per-position gap for one layer, from the cut onward."""

    layer: int
    distances: tuple[float, ...]

    def settles_at(self, epsilon: float) -> int | None:
        """First offset after which the gap never exceeds epsilon. None if it never does."""
        if epsilon < 0:
            raise ValueError(f"epsilon must not be negative, got {epsilon}")
        for index in range(len(self.distances)):
            if all(d <= epsilon for d in self.distances[index:]):
                return index
        return None

    @property
    def peak(self) -> float:
        return max(self.distances, default=0.0)


def _distances(a: torch.Tensor, b: torch.Tensor, start: int) -> tuple[float, ...]:
    out = []
    for offset in range(start, a.shape[-2]):
        out.append(relative_l2(a[..., offset, :], b[..., offset, :]))
    return tuple(out)


def compare(
    left: list[Layer],
    right: list[Layer],
    left_from: int,
    right_from: int,
    use_values: bool = True,
) -> list[Decay]:
    """Offsets line a longer cache up against a shorter one, so the same content is
    compared on both sides.
    """
    if len(left) != len(right):
        raise ValueError(f"{len(left)} left layers, {len(right)} right")
    if not left:
        raise ValueError("no layers to compare")
    for label, stack, offset in (("left", left, left_from), ("right", right, right_from)):
        if not 0 <= offset <= stack[0].length:
            raise ValueError(
                f"{label} offset {offset} but the cache holds {stack[0].length}"
            )

    out = []
    for index, (a, b) in enumerate(zip(left, right)):
        first = a.values if use_values else a.keys
        second = b.values if use_values else b.keys
        span = min(first.shape[-2] - left_from, second.shape[-2] - right_from)
        gaps = tuple(
            relative_l2(first[..., left_from + i, :], second[..., right_from + i, :])
            for i in range(span)
        )
        out.append(Decay(layer=index, distances=gaps))
    return out


def decay(
    spliced: list[Layer], honest: list[Layer], cut_at: int, use_values: bool = True
) -> list[Decay]:
    """Gap per layer from the cut onward, for caches of the same length."""
    if len(spliced) != len(honest):
        raise ValueError(f"{len(spliced)} spliced layers, {len(honest)} honest")
    if not spliced:
        raise ValueError("no layers to compare")
    if not 0 <= cut_at <= spliced[0].length:
        raise ValueError(f"cut at {cut_at} but the cache holds {spliced[0].length}")
    for index, (cut, ref) in enumerate(zip(spliced, honest)):
        if cut.length != ref.length:
            raise ValueError(
                f"layer {index}: {cut.length} spliced entries, {ref.length} honest"
            )
    return compare(spliced, honest, cut_at, cut_at, use_values)


def shadow_length(decays: list[Decay], epsilon: float) -> int | None:
    """How long a shadow must run for every layer to be within epsilon."""
    settles = [d.settles_at(epsilon) for d in decays]
    if any(s is None for s in settles):
        return None
    return max(settles)
