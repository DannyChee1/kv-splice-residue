"""Gate 3: how far past a deletion does its influence reach?

Prefill a prompt, cut a span out of the cache, and compare each layer against an
honest reprefill position by position. The gap at position i tells you whether a
shadow track still has to run there, or whether the main track's entries would
already do.

A shadow only needs to run while the two differ. Both tracks see the same tokens,
so the difference is measurable rather than assumed, which is what gives SCM a
stopping rule instead of a guess. At epsilon = 0 the cache is exact; above it the
shadow stops once the gap stays under epsilon, with an error you measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scm.rotate import relative_l2
from scm.splice import Layer


@dataclass(frozen=True, slots=True)
class Decay:
    """Per-position gap between a spliced cache and an honest one, one layer."""

    layer: int
    distances: tuple[float, ...]

    def settles_at(self, epsilon: float) -> int | None:
        """First offset past the cut after which the gap never exceeds epsilon.

        None means it never settles, so a shadow would have to run to the end.
        """
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


def decay(
    spliced: list[Layer], honest: list[Layer], cut_at: int, use_values: bool = True
) -> list[Decay]:
    """Gap per layer from the cut onward. Values by default, since keys get fixed."""
    if len(spliced) != len(honest):
        raise ValueError(f"{len(spliced)} spliced layers, {len(honest)} honest")
    if not spliced:
        raise ValueError("no layers to compare")
    if not 0 <= cut_at <= spliced[0].length:
        raise ValueError(f"cut at {cut_at} but the cache holds {spliced[0].length}")

    out = []
    for index, (cut, ref) in enumerate(zip(spliced, honest)):
        if cut.length != ref.length:
            raise ValueError(
                f"layer {index}: {cut.length} spliced entries, {ref.length} honest"
            )
        left = cut.values if use_values else cut.keys
        right = ref.values if use_values else ref.keys
        out.append(Decay(layer=index, distances=_distances(left, right, cut_at)))
    return out


def shadow_length(decays: list[Decay], epsilon: float) -> int | None:
    """How long a shadow must run for every layer to be within epsilon.

    None if any layer never settles: at that epsilon the shadow runs to the end.
    """
    settles = [d.settles_at(epsilon) for d in decays]
    if any(s is None for s in settles):
        return None
    return max(settles)
