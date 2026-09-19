"""RoPE re-anchoring: move a cached key as if it had been prefilled elsewhere.

Deleting a span shifts everything after it by delta positions. RoPE composes,
R(a)R(b) = R(a+b), so turning a key cached at position p into the key an honest
prefill would have produced at p+delta is one rotation by delta. The angle only
depends on delta, so every shifted key takes the same rotation.

This is the cheap splice we are measuring against, not something SCM needs. It
fixes where a key sits and nothing else: the value vectors, and the parts of the
key RoPE never touched, still carry whatever the deleted span did to them. That
leftover is the point of the experiment.

Two layouts exist and picking the wrong one is silent, so callers say which:

    interleaved   pairs are (0,1), (2,3), ...      GPT-J, DeepSeek MLA
    half_split    pairs are (j, j + dim/2)         GPT-NeoX, Llama
"""

from __future__ import annotations

import torch

INTERLEAVED = "interleaved"
HALF_SPLIT = "half_split"
LAYOUTS = (INTERLEAVED, HALF_SPLIT)


def _check(dim: int, layout: str) -> None:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if dim % 2:
        raise ValueError(f"rotary dim must be even, got {dim}")


def inv_freq(dim: int, base: float = 10000.0, device=None) -> torch.Tensor:
    """Per-pair angular frequency, one entry for each of the dim/2 pairs."""
    steps = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (steps / dim))


def _pairs(x: torch.Tensor, layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    if layout == INTERLEAVED:
        return x[..., 0::2], x[..., 1::2]
    half = x.shape[-1] // 2
    return x[..., :half], x[..., half:]


def _unpair(even: torch.Tensor, odd: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == HALF_SPLIT:
        return torch.cat((even, odd), dim=-1)
    out = torch.stack((even, odd), dim=-1)
    return out.flatten(-2)


def _spin(x: torch.Tensor, angle: torch.Tensor, layout: str) -> torch.Tensor:
    """Rotate each pair by its own angle, in float32 whatever came in."""
    dtype = x.dtype
    even, odd = _pairs(x.to(torch.float32), layout)
    cos, sin = torch.cos(angle), torch.sin(angle)
    return _unpair(even * cos - odd * sin, even * sin + odd * cos, layout).to(dtype)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    layout: str = INTERLEAVED,
    base: float = 10000.0,
) -> torch.Tensor:
    """Put keys at absolute positions, the way a prefill would. Reference only."""
    _check(x.shape[-1], layout)
    if positions.shape[0] != x.shape[-2]:
        raise ValueError(
            f"got {positions.shape[0]} positions for {x.shape[-2]} keys"
        )
    angle = positions.to(torch.float32)[:, None] * inv_freq(
        x.shape[-1], base, x.device
    )
    return _spin(x, angle, layout)


def rotate(
    x: torch.Tensor,
    delta: int,
    layout: str = INTERLEAVED,
    base: float = 10000.0,
) -> torch.Tensor:
    """Re-anchor already-placed keys by delta positions.

    One angle for the whole tensor, since R(delta) does not depend on where the
    key started. Negative delta moves keys earlier, which is the deletion case.
    """
    _check(x.shape[-1], layout)
    angle = float(delta) * inv_freq(x.shape[-1], base, x.device)
    return _spin(x, angle, layout)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Leyline reports rel-L2, so we report it the same way."""
    scale = torch.linalg.vector_norm(expected.to(torch.float32))
    if scale == 0:
        raise ValueError("expected tensor is all zeros, rel-L2 undefined")
    diff = torch.linalg.vector_norm((actual - expected).to(torch.float32))
    return (diff / scale).item()
