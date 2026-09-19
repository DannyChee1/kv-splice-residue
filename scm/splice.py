"""Cut a span out of a KV cache without re-prefilling what follows.

A reference implementation of the positional splice Leyline specifies, written
in plain PyTorch rather than their fused MLA kernel. It is here to reproduce the
behavior, not the performance. Drop the span's entries, then re-anchor every key
after it by rotating down `end - start` positions so the arithmetic lines up.

What the rotation does not reach is what we are measuring. Values carry over
unchanged, and on MLA so does the part of the key RoPE never covers. Both were
computed while the model was attending to the span now being removed. Leyline
says so plainly and means it: the contract is positional, not informational, and
skipping that recompute is the cost the splice exists to avoid.

An honest prefill of the shortened prompt carries none of that history, which is
what SCM produces and what makes the two tellable apart. `scm.agreement` does
the telling.

Caches are [..., seq, head_dim] with sequence second from last, matching the
layout HuggingFace hands back.
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
    """Drop keys [start, end) and re-anchor the rest by the shift.

    `rope_dim` is the width of the trailing slice RoPE actually covers, for MLA
    where the rest of the key is position-free. None means the whole key.
    """
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
    """Drop values [start, end). The survivors are copied over as they are."""
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
    """Cut a span from one layer.

    `mla` swaps which tensor gets rotated. HuggingFace's MLA does not cache keys
    and values at all: the `keys` slot holds the compressed KV latent and the
    `values` slot holds k_pe, the only part RoPE ever touched. So the rotation
    belongs on `values`, whole, and the latent is carried across untouched.

    That makes the leftover starker than on ordinary attention. The latent is
    position-free by construction, so no rotation could repair it even in
    principle, and it holds everything the model took from the span being cut.
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
    """Apply the same cut to every layer. One directive, one span, whole stack."""
    if not cache:
        raise SpliceError("cache has no layers")
    lengths = {layer.length for layer in cache}
    if len(lengths) != 1:
        raise SpliceError(f"layers disagree on length: {sorted(lengths)}")
    return [
        splice_layer(la, start, end, layout, rope_dim, base, mla, freqs)
        for la in cache
    ]
