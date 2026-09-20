"""Leyline's positional splice, in plain PyTorch rather than their fused kernel.

Values and the position-free part of the key carry over untouched. That leftover
is what everything downstream is trying to measure.

Caches are [..., seq, head_dim], as HuggingFace hands them back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scm.rotate import INTERLEAVED, rotate

SEQ = -2


class SpliceError(ValueError):
    pass


def _check_span(length: int, start: int, end: int) -> None:
    if start < 0 or end < start:
        raise SpliceError(f"bad span ({start}, {end})")
    if end > length:
        raise SpliceError(f"span ends at {end}, cache holds {length}")


def splice_keys(
    keys: torch.Tensor,
    start: int,
    end: int,
    layout: str = INTERLEAVED,
    rope_dim: int | None = None,
    base: float = 10000.0,
    freqs: torch.Tensor | None = None,
) -> torch.Tensor:
    """`rope_dim` is the trailing slice RoPE covers, for MLA. None means the whole key."""
    _check_span(keys.shape[SEQ], start, end)
    delta = -(end - start)
    if delta == 0:
        return keys

    head = keys[..., :start, :]
    tail = keys[..., end:, :]
    if rope_dim is None:
        tail = rotate(tail, delta, layout, base, freqs)
    else:
        if not 0 < rope_dim <= keys.shape[-1]:
            raise SpliceError(
                f"rope_dim {rope_dim} does not fit a key of {keys.shape[-1]}"
            )
        plain, spun = tail[..., :-rope_dim], tail[..., -rope_dim:]
        tail = torch.cat(
            (plain, rotate(spun, delta, layout, base, freqs)), dim=-1
        )
    return torch.cat((head, tail), dim=SEQ)


def splice_values(values: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Drop values [start, end). Survivors are copied as they are."""
    _check_span(values.shape[SEQ], start, end)
    if start == end:
        return values
    return torch.cat((values[..., :start, :], values[..., end:, :]), dim=SEQ)


@dataclass(frozen=True, slots=True)
class Layer:
    keys: torch.Tensor
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.keys.shape[SEQ] != self.values.shape[SEQ]:
            raise SpliceError(
                f"{self.keys.shape[SEQ]} keys but {self.values.shape[SEQ]} values"
            )

    @property
    def length(self) -> int:
        return self.keys.shape[SEQ]


def splice_layer(
    layer: Layer,
    start: int,
    end: int,
    layout: str = INTERLEAVED,
    rope_dim: int | None = None,
    base: float = 10000.0,
    mla: bool = False,
    freqs: torch.Tensor | None = None,
) -> Layer:
    """MLA's `keys` slot holds the latent and `values` holds k_pe, so on MLA the
    rotation goes on `values` and the latent is carried over untouched.
    """
    if mla:
        return Layer(
            keys=splice_values(layer.keys, start, end),
            values=splice_keys(
                layer.values, start, end, layout, None, base, freqs
            ),
        )
    return Layer(
        keys=splice_keys(layer.keys, start, end, layout, rope_dim, base, freqs),
        values=splice_values(layer.values, start, end),
    )


def splice_cache(
    cache: list[Layer],
    start: int,
    end: int,
    layout: str = INTERLEAVED,
    rope_dim: int | None = None,
    base: float = 10000.0,
    mla: bool = False,
    freqs: torch.Tensor | None = None,
) -> list[Layer]:
    """Apply the same cut to every layer."""
    if not cache:
        raise SpliceError("cache has no layers")
    lengths = {layer.length for layer in cache}
    if len(lengths) != 1:
        raise SpliceError(f"layers disagree on length: {sorted(lengths)}")
    return [
        splice_layer(la, start, end, layout, rope_dim, base, mla, freqs)
        for la in cache
    ]
